"""Deterministic playback; exposes task commands without the legacy keyboard code."""
import argparse
import json
import sys
import traceback
from pathlib import Path
import bootstrap
from isaaclab.app import AppLauncher
from playback_config import resolve_playback_args, restore_runtime_settings

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=("stand", "height", "velocity", "jump"), default=None)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--allow_auxiliary_change", action="store_true",
                    help="Explicitly evaluate matching v2 weights with changed auxiliary joint control")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--robot_spec", type=str)
parser.add_argument("--strategy", choices=("baseline", "legacy_ppo"), default=None)
parser.add_argument("--actor_history", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--motor_curves", type=Path)
parser.add_argument("--vx", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--command_height", type=float, help="Robot target height in meters (not the render/window height)")
parser.add_argument("--jump_at", type=float, default=2.0, help="Seconds until one jump request")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
try:
    saved_schema = resolve_playback_args(args)
    if args.num_envs < 1 or args.steps < 1:
        raise ValueError("num_envs and steps must be positive")
except (ValueError, OSError) as exc:
    parser.error(str(exc))
print(f"[INFO] Playback: task={args.task}, strategy={args.strategy}, actor_history={args.actor_history}, seed={args.seed}", flush=True)
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import wheel_legged_lab
from wheel_legged_lab.config import WheelLeggedEnvCfg
from wheel_legged_lab.settings import settings_for
from wheel_legged_lab.ppo import WheelLeggedPPORunnerCfg
from wheel_legged_lab.checkpoints import validate_checkpoint

def main():
    settings = settings_for(args.task, args.robot_spec, args.strategy)
    restore_runtime_settings(settings, saved_schema)
    if args.motor_curves:
        settings.motors.torque_speed_curves = json.loads(args.motor_curves.read_text())
    settings.observations.noise_enabled = False
    settings.jump.automatic = False
    cfg = WheelLeggedEnvCfg(settings=settings)
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, args.seed
    env = RslRlVecEnvWrapper(gym.make(f"WheelLegged-{args.task.title()}-Direct-v0", cfg=cfg))
    try:
        u = env.unwrapped
        validate_checkpoint(args.checkpoint, u.observation_schema, allow_auxiliary_change=args.allow_auxiliary_change)
        runner = OnPolicyRunner(env, WheelLeggedPPORunnerCfg().to_dict(), log_dir=None, device=args.device)
        runner.load(str(args.checkpoint), load_optimizer=False, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        print(f"[INFO] Policy loaded. Replaying {args.steps} steps.", flush=True)
        with torch.inference_mode():
            obs, _ = env.reset()
            u.set_play_velocity_command(vx=args.vx, vy=0.0, wz=args.wz)
            if args.command_height is not None:
                u.set_height_command(args.command_height)
            obs = env.get_observations()
            for step in range(args.steps):
                if not app.is_running():
                    break
                if args.task == "jump" and step == round(args.jump_at / u.step_dt):
                    u.request_jump_takeoff()
                obs, _, _, _ = env.step(policy(obs))
    finally:
        env.close()

try:
    main()
except BaseException:
    # Kit fast shutdown can terminate Python before an unwinding exception is
    # displayed. Print and flush the actual failure before closing the app.
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    app.close()
