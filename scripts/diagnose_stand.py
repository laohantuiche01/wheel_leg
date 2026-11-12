"""Measure a standing checkpoint's tilt and left/right leg geometry; no training."""
import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
import bootstrap
from isaaclab.app import AppLauncher
from playback_config import resolve_playback_args, restore_runtime_settings

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--allow_auxiliary_change", action="store_true")
parser.add_argument("--strategy", choices=("baseline", "legacy_ppo"), default=None)
parser.add_argument("--actor_history", type=int, default=None)
parser.add_argument("--motor_curves", type=Path)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.task, args.robot_spec = "stand", None
try:
    saved_schema = resolve_playback_args(args)
except (ValueError, OSError) as exc:
    parser.error(str(exc))
if args.steps < 1 or args.num_envs < 1:
    parser.error("steps and num_envs must be positive")
# Reserve new output paths before simulation; never overwrite an earlier report.
args.output_dir.mkdir(parents=True, exist_ok=False)
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import wheel_legged_lab
from wheel_legged_lab.config import WheelLeggedEnvCfg
from wheel_legged_lab.settings import settings_for
from wheel_legged_lab.ppo import WheelLeggedPPORunnerCfg
from wheel_legged_lab.checkpoints import validate_checkpoint
from wheel_legged_lab.components.state import wheel_fore_aft_difference


def main():
    settings = settings_for("stand", args.robot_spec, args.strategy)
    restore_runtime_settings(settings, saved_schema)
    if args.motor_curves:
        settings.motors.torque_speed_curves = json.loads(args.motor_curves.read_text())
    settings.observations.noise_enabled = False
    cfg = WheelLeggedEnvCfg(settings=settings)
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, args.seed
    env = RslRlVecEnvWrapper(gym.make("WheelLegged-Stand-Direct-v0", cfg=cfg))
    try:
        u = env.unwrapped
        validate_checkpoint(args.checkpoint, u.observation_schema, allow_auxiliary_change=args.allow_auxiliary_change)
        agent = WheelLeggedPPORunnerCfg()
        agent.device, agent.seed = args.device, args.seed
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=args.device)
        runner.load(str(args.checkpoint), load_optimizer=False, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        robot, binding = u.robot, u.binding
        hips = [robot.body_names.index(f"{side}_thigh_link") for side in ("left", "right")]
        gimbal = [robot.joint_names.index(f"gimbal_{axis}_joint") for axis in ("yaw", "pitch")]
        non_wheel_sensor_ids = [i for i in range(len(u.contact_sensor.body_names)) if i not in binding.wheel_sensor_ids]
        # Read the actual yaw joint axis in its parent body's frame, rather than
        # assuming that the USD 'axis=X' token denotes world X.
        from pxr import Gf, Usd, UsdPhysics
        stage = Usd.Stage.Open(u.description.usd)
        joint = next(UsdPhysics.Joint(p) for p in stage.Traverse() if p.GetName() == "gimbal_yaw_joint")
        parent = joint.GetBody0Rel().GetTargets()[0].name
        axis_index = {"X": 0, "Y": 1, "Z": 2}[joint.GetPrim().GetAttribute("physics:axis").Get()]
        axis = Gf.Vec3d(0)
        axis[axis_index] = 1
        axis = Gf.Rotation(Gf.Quatd(joint.GetLocalRot0Attr().Get())).TransformDir(axis)
        parent_id = robot.body_names.index(parent)
        axis_b = torch.tensor(list(axis), device=u.device, dtype=torch.float32).repeat(u.num_envs, 1)
        names = ["episode_age_s", "roll_deg", "pitch_deg", "yaw_axis_tilt_deg", "height_m",
                 "hip_left_minus_right_z_m", "wheel_left_minus_right_z_m",
                 "left_vertical_leg_m", "right_vertical_leg_m", "left_contact", "right_contact",
                 "left_normal_force_N", "right_normal_force_N", "gimbal_yaw_rad", "gimbal_pitch_rad",
                 "left_spring_m", "right_spring_m", "wheel_left_minus_right_body_x_m", "non_wheel_contact_count"]
        for block in ("q_rad", "target", "action", "effort_Nm"):
            names.extend(f"{name}/{block}" for name in u.description.action_names)
        rows, done_count, body_contacts = [], 0, []
        with torch.inference_mode(), (args.output_dir / "trajectory.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["step", "env", *names])
            obs, _ = env.reset()
            u.set_play_velocity_command(vx=0, vy=0, wz=0)
            obs = env.get_observations()
            for step in range(args.steps):
                if not app.is_running():
                    break
                obs, _, dones, _ = env.step(policy(obs))
                done_count += int(dones.sum())
                d = robot.data
                gravity = d.projected_gravity_b
                roll = torch.atan2(-gravity[:, 1], -gravity[:, 2]).rad2deg()
                pitch = torch.atan2(gravity[:, 0], gravity[:, 1:].norm(dim=1)).rad2deg()
                yaw_axis_w = quat_apply(d.body_quat_w[:, parent_id], axis_b)
                tilt = torch.acos(yaw_axis_w[:, 2].clamp(-1, 1)).rad2deg()
                hip_z = d.body_pos_w[:, hips, 2]
                wheel_z = d.body_pos_w[:, binding.wheel_body_ids, 2]
                contacts = u.reader.read().wheel_contact
                forces = u.contact_sensor.data.net_forces_w[:, binding.wheel_sensor_ids, 2]
                all_contacts = u.contact_sensor.data.net_forces_w.norm(dim=-1) > settings.contact_force_threshold
                physical_action = u.motors.applied * u.motors.directions
                targets = []
                for group, _, columns, ids in u.motors.groups:
                    target = physical_action[:, columns] * group.action_scale
                    if group.mode == "position":
                        target = (d.default_joint_pos[:, ids] + target).clamp(
                            d.soft_joint_pos_limits[:, ids, 0], d.soft_joint_pos_limits[:, ids, 1])
                    elif group.mode == "velocity":
                        target = target.clamp(-group.speed_limit, group.speed_limit)
                    targets.append(target)
                values = torch.cat((
                    (u.episode_length_buf * u.step_dt)[:, None], roll[:, None], pitch[:, None], tilt[:, None],
                    (d.root_pos_w[:, 2] - u.scene.env_origins[:, 2])[:, None],
                    (hip_z[:, 0] - hip_z[:, 1])[:, None], (wheel_z[:, 0] - wheel_z[:, 1])[:, None],
                    hip_z - wheel_z, contacts.float(), forces, d.joint_pos[:, gimbal],
                    d.joint_pos[:, binding.spring_ids],
                    wheel_fore_aft_difference(d.root_quat_w, d.body_pos_w[:, binding.wheel_body_ids])[:, None],
                    all_contacts[:, non_wheel_sensor_ids].sum(dim=1, keepdim=True),
                    d.joint_pos[:, binding.action_ids],
                    torch.cat(targets, dim=1), u.motors.actions,
                    u.motors.effort * u.motors.directions), dim=1).cpu()
                for index, values_row in enumerate(values.tolist()):
                    writer.writerow([step, index, *values_row])
                # Drop reset and first two seconds, so steady metrics are not
                # dominated by the intentional zero-pose initialization.
                keep = (u.episode_length_buf * u.step_dt >= 2) & ~dones.bool()
                rows.append(values[keep.cpu()])
                body_contacts.append(all_contacts[keep].float().cpu())
        samples = torch.cat(rows) if rows else torch.empty(0, len(names))
        metrics = {name: {"mean": float(samples[:, i].mean()),
                          "rms": float(samples[:, i].square().mean().sqrt()),
                          "min": float(samples[:, i].min()), "max": float(samples[:, i].max())}
                   for i, name in enumerate(names)} if len(samples) else {}
        report = dict(checkpoint=str(args.checkpoint.resolve()), task="stand", seed=args.seed,
            num_envs=args.num_envs, requested_steps=args.steps, steady_samples=len(samples),
            resets=done_count, yaw_axis_parent=parent, yaw_axis_parent_frame=list(axis),
            metrics=metrics, strategy=args.strategy, auxiliary_holds=u.observation_schema["auxiliary_holds"],
            allow_auxiliary_change=args.allow_auxiliary_change, training_started=False)
        if len(samples):
            fractions = torch.cat(body_contacts).mean(dim=0).tolist()
            report["body_contact_fractions"] = dict(zip(u.contact_sensor.body_names, fractions))
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
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
