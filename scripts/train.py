"""Train the new composition task. Execute only after runtime validation is approved."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import bootstrap
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=("stand", "height", "velocity", "jump"), default="stand")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--max_iterations", type=int, default=3000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--robot_spec", type=str)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--mass_randomization", action="store_true")
parser.add_argument("--pushes", action="store_true")
parser.add_argument("--log_root", type=Path, default=bootstrap.ROOT / "logs")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.io import dump_yaml
from rsl_rl.runners import OnPolicyRunner
import wheel_legged_lab
from wheel_legged_lab.config import WheelLeggedEnvCfg
from wheel_legged_lab.settings import settings_for
from wheel_legged_lab.ppo import WheelLeggedPPORunnerCfg
from wheel_legged_lab.checkpoints import save_schema, validate_checkpoint

def main():
    settings = settings_for(args.task, args.robot_spec)
    settings.mass.enabled = args.mass_randomization
    settings.force.enabled = args.pushes
    cfg = WheelLeggedEnvCfg(settings=settings)
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, args.seed
    agent = WheelLeggedPPORunnerCfg()
    agent.device, agent.seed, agent.max_iterations = args.device, args.seed, args.max_iterations
    if args.resume and not args.checkpoint:
        raise ValueError("--resume requires --checkpoint")
    log = args.log_root / args.task / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log.mkdir(parents=True, exist_ok=False)
    env = RslRlVecEnvWrapper(gym.make(f"WheelLegged-{args.task.title()}-Direct-v0", cfg=cfg))
    try:
        schema = env.unwrapped.observation_schema
        save_schema(log, schema)
        dump_yaml(str(log / "env.yaml"), cfg)
        dump_yaml(str(log / "agent.yaml"), agent)
        (log / "launch.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(log), device=args.device)
        if args.checkpoint:
            validate_checkpoint(args.checkpoint, schema)
            runner.load(str(args.checkpoint), load_optimizer=args.resume, map_location=args.device)
            if args.resume:
                runner.current_learning_iteration += 1
            else:
                runner.current_learning_iteration = 0
                with torch.no_grad():
                    runner.alg.policy.std.fill_(agent.policy.init_noise_std)
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    finally:
        env.close()

try:
    main()
finally:
    app.close()
