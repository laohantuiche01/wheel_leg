"""Compare walk checkpoints; pure-translation configurations exclude all turns."""
import argparse
import json
from pathlib import Path
import sys
import traceback
import bootstrap
from isaaclab.app import AppLauncher
from playback_config import resolve_playback_args, restore_runtime_settings

CASES = [
    ("stop", 0., 0.), ("forward_02", .2, 0.), ("reverse_02", -.2, 0.),
    ("forward_05", .5, 0.), ("reverse_05", -.5, 0.),
    ("forward_10", 1., 0.), ("reverse_10", -1., 0.),
    ("left", .5, .6), ("right", .5, -.6),
    ("reverse_left", -.5, .6), ("reverse_right", -.5, -.6),
    ("left_fast_yaw", .5, 1.2), ("right_fast_yaw", .5, -1.2),
]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--repeats", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--extended_range", action="store_true",
                    help="Also test signed 1.5/2 m/s and yaw rates up to 2 rad/s")
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--task", default=None)
parser.add_argument("--strategy", default=None)
parser.add_argument("--actor_history", type=int, default=None)
parser.add_argument("--robot_spec", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.extended_range:
    CASES += [
        ("forward_15", 1.5, 0.), ("reverse_15", -1.5, 0.),
        ("forward_20", 2., 0.), ("reverse_20", -2., 0.),
        ("left_yaw_20", .5, 2.), ("right_yaw_20", .5, -2.),
        ("fast_left", 1., 2.), ("fast_right", 1., -2.),
        ("reverse_fast_left", -1., 2.), ("reverse_fast_right", -1., -2.),
        ("corner_left", 2., 2.), ("corner_right", 2., -2.),
        ("reverse_corner_left", -2., 2.), ("reverse_corner_right", -2., -2.),
    ]
try:
    args.checkpoint = args.checkpoints[0]
    schema = resolve_playback_args(args)
    if args.task != "walk" or min(args.steps, args.repeats) < 1:
        raise ValueError("Use matching walk checkpoints and positive steps/repeats")
    translation_only = tuple(schema.get("motion_task", {}).get("commands", {}).get("wz", (0., 0.))) == (0., 0.)
    if translation_only:
        CASES = [case for case in CASES if case[2] == 0.]
    args.checkpoints = [p.expanduser().resolve() for p in args.checkpoints]
    if not all(p.is_file() for p in args.checkpoints):
        raise ValueError("A checkpoint is missing")
    args.output_dir.mkdir(parents=True, exist_ok=False)
except (ValueError, OSError) as exc:
    parser.error(str(exc))
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import wheel_legged_lab
from wheel_legged_lab.settings import settings_for
from wheel_legged_lab.config import WheelLeggedEnvCfg
from wheel_legged_lab.ppo import WheelLeggedPPORunnerCfg
from wheel_legged_lab.checkpoints import validate_checkpoint
from wheel_legged_lab.components.state import body_tilt_angles


def main():
    settings = settings_for(args.task, args.robot_spec, args.strategy)
    restore_runtime_settings(settings, schema)
    cfg = WheelLeggedEnvCfg(settings=settings)
    cfg.scene.num_envs = len(CASES) * args.repeats
    cfg.sim.device, cfg.seed = args.device, args.seed
    env = RslRlVecEnvWrapper(gym.make("WheelLegged-Walk-Direct-v0", cfg=cfg))
    try:
        u = env.unwrapped
        runner = OnPolicyRunner(env, WheelLeggedPPORunnerCfg().to_dict(), log_dir=None, device=args.device)
        policy = runner.get_inference_policy(device=args.device)
        columns = ["command_vx", "command_wz", "actual_vx", "actual_wz", "v_error", "w_error",
                   "roll_deg", "pitch_deg", "stagger_m", "both_contact", "nonwheel_contact",
                   "height_m", "done", "finite", "age_s", "steady", "terminated", "timeout"]
        results = []
        for index, path in enumerate(args.checkpoints):
            validate_checkpoint(path, u.observation_schema)
            runner.load(str(path), load_optimizer=False, map_location=args.device)
            # Reproduce initial yaw, motor-delay and subsequent reset samples.
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            with torch.inference_mode():
                obs, _ = env.reset()
                warmup = torch.empty(u.num_envs, device=u.device)
                for case, (_, vx, wz) in enumerate(CASES):
                    ids = torch.arange(case * args.repeats, (case + 1) * args.repeats, device=u.device)
                    u.set_play_velocity_command(vx=vx, vy=0., wz=wz, env_ids=ids)
                    warmup[ids] = (settings.commands.settle_s +
                        max(abs(vx) / settings.commands.linear_accel, abs(wz) / settings.commands.yaw_accel) + 2.)
                obs = env.get_observations()
                frames = []
                for step in range(args.steps):
                    if not app.is_running():
                        raise RuntimeError("Simulator closed before evaluation completed")
                    command = u.commands.velocity.clone()
                    obs, _, dones, _ = env.step(policy(obs))
                    state = u.reader.read()
                    roll, pitch = body_tilt_angles(state.gravity)
                    age = u.episode_length_buf * u.step_dt
                    finite = state.finite & torch.isfinite(state.axle_linear_velocity_heading).all(dim=1)
                    steady = (age >= warmup) & ~dones.bool() & finite
                    frames.append(torch.stack((command[:, 0], command[:, 2],
                        state.axle_linear_velocity_heading[:, 0], state.yaw_rate_world,
                        command[:, 0] - state.axle_linear_velocity_heading[:, 0],
                        command[:, 2] - state.yaw_rate_world, roll.rad2deg(), pitch.rad2deg(),
                        state.wheel_fore_aft_difference, state.wheel_contact.all(dim=1).float(),
                        state.non_wheel_contact.float(), state.height, dones.float(), finite.float(),
                        age, steady.float(), u.reset_terminated.float(), u.reset_time_outs.float()), dim=-1).cpu())
                trajectory = torch.stack(frames).numpy()
            cases = []
            for case, (name, vx, wz) in enumerate(CASES):
                all_rows = trajectory[:, case * args.repeats:(case + 1) * args.repeats].reshape(-1, len(columns))
                rows = all_rows[all_rows[:, 15] > .5]
                result = dict(name=name, vx=vx, wz=wz, resets=int(all_rows[:, 12].sum()),
                    terminations=int(all_rows[:, 16].sum()), timeouts=int(all_rows[:, 17].sum()),
                    nonfinite_states=int((all_rows[:, 13] < .5).sum()), steady_samples=len(rows))
                if len(rows):
                    def rms(column):
                        return float(np.sqrt(np.mean(np.square(rows[:, column]))))
                    result.update(actual_v_mean=float(rows[:, 2].mean()), actual_w_mean=float(rows[:, 3].mean()),
                        v_bias=float(rows[:, 4].mean()), v_mae=float(np.abs(rows[:, 4]).mean()),
                        v_rmse=rms(4), v_absolute_error_p95=float(np.quantile(np.abs(rows[:, 4]), .95)),
                        w_rmse=rms(5), roll_rms_deg=rms(6), pitch_rms_deg=rms(7), stagger_rms_m=rms(8),
                        both_contact_fraction=float(rows[:, 9].mean()), nonwheel_contact_fraction=float(rows[:, 10].mean()))
                cases.append(result)
            valid = [c for c in cases if c["steady_samples"]]
            summary = dict(checkpoint=str(path), completed_steps=args.steps, cases=cases,
                total_resets=sum(c["resets"] for c in cases),
                total_terminations=sum(c["terminations"] for c in cases),
                total_timeouts=sum(c["timeouts"] for c in cases),
                cases_without_steady_samples=len(cases) - len(valid),
                nonfinite_states=sum(c["nonfinite_states"] for c in cases),
                mean_case_v_rmse=float(np.mean([c["v_rmse"] for c in valid])) if valid else None,
                mean_case_w_rmse=float(np.mean([c["w_rmse"] for c in valid])) if valid else None,
                worst_case_pitch_rms_deg=max((c["pitch_rms_deg"] for c in valid), default=None))
            label = f"{index:02d}_{path.stem}"
            np.savez_compressed(args.output_dir / f"{label}_trajectory.npz", values=trajectory, columns=columns)
            (args.output_dir / f"{label}.json").write_text(json.dumps(summary, indent=2) + "\n")
            results.append(summary)
            print(json.dumps({k: v for k, v in summary.items() if k != "cases"}), flush=True)
            report = dict(seed=args.seed, repeats=args.repeats, step_dt=u.step_dt,
                extended_range=args.extended_range, case_count=len(CASES),
                translation_only=translation_only,
                steps=args.steps, steady_window="2 seconds after the command ramp completes; exclude reset/nonfinite rows",
                scope=f"Flat ground; {len(CASES)} constant signed-speed/turn commands; randomized yaw and motor delays; deterministic policy",
                note="Reset counts cover all steps. Reported errors are conditional on steady samples. Repeats are initializations under one RNG seed.",
                results=results)
            (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    app.close()
