"""Deterministic playback; exposes task commands without the legacy keyboard code."""
import argparse
from pathlib import Path
import bootstrap
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=("stand", "height", "velocity", "jump"), default="velocity")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--allow_auxiliary_change", action="store_true",
                    help="Explicitly evaluate matching v2 weights with changed auxiliary joint control")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--robot_spec", type=str)
parser.add_argument("--vx", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--command_height", type=float, help="Robot target height in meters (not the render/window height)")
parser.add_argument("--jump_at", type=float, default=2.0, help="Seconds until one jump request")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
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
    settings = settings_for(args.task, args.robot_spec)
    settings.observations.noise_enabled = False
    settings.jump.automatic = False
    cfg = WheelLeggedEnvCfg(settings=settings)
    cfg.scene.num_envs, cfg.sim.device = args.num_envs, args.device
    env = RslRlVecEnvWrapper(gym.make(f"WheelLegged-{args.task.title()}-Direct-v0", cfg=cfg))
    try:
        u = env.unwrapped
        validate_checkpoint(args.checkpoint, u.observation_schema, allow_auxiliary_change=args.allow_auxiliary_change)
        runner = OnPolicyRunner(env, WheelLeggedPPORunnerCfg().to_dict(), log_dir=None, device=args.device)
        runner.load(str(args.checkpoint), load_optimizer=False, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
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
finally:
    app.close()
