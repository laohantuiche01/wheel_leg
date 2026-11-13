"""Test spin tracking, axle drift and gimbal world heading at signed yaw rates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
import bootstrap
from isaaclab.app import AppLauncher
from playback_config import resolve_playback_args, restore_runtime_settings

RATES = (0., 1., -1., 2., -2., 3., -3.)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--repeats", type=int, default=4)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--gimbal_heading", type=float, default=None)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--task", default=None)
parser.add_argument("--strategy", default=None)
parser.add_argument("--actor_history", type=int, default=None)
parser.add_argument("--robot_spec", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
try:
    schema = resolve_playback_args(args)
    if args.task != "spin" or min(args.steps, args.repeats) < 1:
        raise ValueError("Use a spin checkpoint and positive steps/repeats")
    saved_range = schema.get("motion_task", {}).get("commands", {}).get("wz", (-3., 3.))
    RATES = tuple(rate for rate in RATES if saved_range[0] <= rate <= saved_range[1])
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
from wheel_legged_lab.components.locomotion import heading_from_quaternion, wrap_angle


def main():
    settings = settings_for(args.task, args.robot_spec, args.strategy)
    restore_runtime_settings(settings, schema)
    cfg = WheelLeggedEnvCfg(settings=settings)
    cfg.scene.num_envs = len(RATES) * args.repeats
    cfg.sim.device, cfg.seed = args.device, args.seed
    env = RslRlVecEnvWrapper(gym.make("WheelLegged-Spin-Direct-v0", cfg=cfg))
    try:
        u = env.unwrapped
        validate_checkpoint(args.checkpoint, u.observation_schema)
        runner = OnPolicyRunner(env, WheelLeggedPPORunnerCfg().to_dict(), log_dir=None, device=args.device)
        runner.load(str(args.checkpoint), load_optimizer=False, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        columns = ["age_s", "command_wz", "actual_wz", "yaw_error", "axle_vx", "axle_vy",
            "axle_speed", "roll_deg", "pitch_deg", "gimbal_error_deg", "gimbal_world_wz",
            "gimbal_joint_velocity", "gimbal_effort", "both_contact", "nonwheel_contact",
            "axle_displacement_m", "terminated", "timeout", "finite", "active", "steady",
            "command_reached", "signed_angle_rad", "height_m"]
        with torch.inference_mode():
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            obs, _ = env.reset()
            target = torch.empty(u.num_envs, device=u.device)
            warmup = torch.empty_like(target)
            for case, rate in enumerate(RATES):
                ids = torch.arange(case * args.repeats, (case + 1) * args.repeats, device=u.device)
                u.set_play_velocity_command(vx=0., vy=0., wz=rate, env_ids=ids)
                target[ids] = rate
                warmup[ids] = settings.commands.settle_s + abs(rate) / settings.commands.yaw_accel + 2.
            if args.gimbal_heading is not None:
                u.set_gimbal_heading(args.gimbal_heading)
            obs = env.get_observations()
            anchor = u.reader.read().axle_position_world[:, :2].clone()
            previous_heading = heading_from_quaternion(u.robot.data.root_quat_w)
            accumulated_angle = torch.zeros_like(target)
            first_episode_duration = torch.zeros_like(target)
            first_episode_ended = torch.zeros(u.num_envs, device=u.device, dtype=torch.bool)
            frames = []
            for step in range(args.steps):
                if not app.is_running():
                    raise RuntimeError("Simulation closed before the requested test completed")
                command = u.commands.velocity[:, 2].clone()
                old_age = u.episode_length_buf * u.step_dt
                obs, _, dones, _ = env.step(policy(obs))
                state = u.reader.read()
                age = u.episode_length_buf * u.step_dt
                roll, pitch = body_tilt_angles(state.gravity)
                heading = heading_from_quaternion(u.robot.data.root_quat_w)
                accumulated_angle += torch.where(dones.bool(), 0., wrap_angle(heading - previous_heading))
                previous_heading = heading
                first_episode_duration[~first_episode_ended] = old_age[~first_episode_ended] + u.step_dt
                first_episode_ended |= dones.bool()
                anchor[dones.bool()] = state.axle_position_world[dones.bool(), :2]
                speed = state.axle_linear_velocity_heading[:, :2].norm(dim=1)
                finite = state.finite & torch.isfinite(state.gimbal_heading_error) & torch.isfinite(speed)
                valid = finite & ~dones.bool()
                active = valid & (age >= settings.commands.settle_s)
                steady = valid & (age >= warmup)
                reached = active & ((command - target).abs() < .03)
                frames.append(torch.stack((age, command, state.yaw_rate_world,
                    command - state.yaw_rate_world, state.axle_linear_velocity_heading[:, 0],
                    state.axle_linear_velocity_heading[:, 1], speed, roll.rad2deg(), pitch.rad2deg(),
                    state.gimbal_heading_error.rad2deg(), state.gimbal_world_yaw_rate,
                    u.robot.data.joint_vel[:, u.gimbal_heading.joint_id], u.gimbal_heading.effort[:, 0],
                    state.wheel_contact.all(dim=1).float(), state.non_wheel_contact.float(),
                    (state.axle_position_world[:, :2] - anchor).norm(dim=1),
                    u.reset_terminated.float(), u.reset_time_outs.float(), finite.float(),
                    active.float(), steady.float(), reached.float(), accumulated_angle, state.height), dim=-1).cpu())
            trajectory = torch.stack(frames).numpy()
        results = []
        for case, rate in enumerate(RATES):
            block = trajectory[:, case * args.repeats:(case + 1) * args.repeats]
            rows = block.reshape(-1, len(columns))
            def metrics(mask):
                selected = rows[mask]
                if not len(selected):
                    return None
                def rms(index):
                    return float(np.sqrt(np.mean(np.square(selected[:, index]))))
                spinning = selected[(np.sign(rate) * selected[:, 2] >= max(.5, .5 * abs(rate)))] if rate else np.empty((0, len(columns)))
                return dict(samples=len(selected), actual_wz_mean=float(selected[:, 2].mean()),
                    actual_wz_rms=rms(2), yaw_error_rms=rms(3), axle_speed_rms=rms(6),
                    roll_rms_deg=rms(7), pitch_rms_deg=rms(8), gimbal_heading_error_rms_deg=rms(9),
                    gimbal_heading_error_p95_deg=float(np.quantile(np.abs(selected[:, 9]), .95)),
                    gimbal_world_wz_rms=rms(10), gimbal_joint_velocity_mean=float(selected[:, 11].mean()),
                    gimbal_effort_saturation_fraction=float((np.abs(selected[:, 12]) >= .99 * settings.spin.effort_limit).mean()),
                    both_contact_fraction=float(selected[:, 13].mean()),
                    axle_displacement_max_m=float(selected[:, 15].max()),
                    achieved_half_requested_spin_samples=len(spinning),
                    gimbal_heading_rms_when_spinning_deg=float(np.sqrt(np.mean(spinning[:, 9] ** 2))) if len(spinning) else None)
            results.append(dict(requested_wz=rate, terminations=int(rows[:, 16].sum()),
                timeouts=int(rows[:, 17].sum()), nonfinite_states=int((rows[:, 18] < .5).sum()),
                first_episode_duration_s=first_episode_duration[case * args.repeats:(case + 1) * args.repeats].cpu().tolist(),
                requested_command_samples=int((rows[:, 21] > .5).sum()),
                accumulated_signed_yaw_per_initialization_rad=block[-1, :, 22].tolist(),
                active=metrics(rows[:, 19] > .5), steady=metrics(rows[:, 20] > .5)))
        report = dict(checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            task=args.task, strategy=args.strategy, seed=args.seed, repeats=args.repeats,
            steps=args.steps, step_dt=u.step_dt, num_envs=u.num_envs, gimbal_heading=args.gimbal_heading,
            tested_rates=RATES,
            cases=results, total_terminations=sum(c["terminations"] for c in results),
            total_timeouts=sum(c["timeouts"] for c in results),
            note="Active metrics exclude reset rows and the initial settle time. Steady metrics wait 2 s beyond the nominal ramp. Small heading error without achieved chassis spin does not prove dynamic gimbal stabilization.",
            training_started=False)
        np.savez_compressed(args.output_dir / "trajectory.npz", values=trajectory, columns=columns)
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
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
