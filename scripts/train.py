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
parser.add_argument("--strategy", choices=("baseline", "legacy_ppo"), default="baseline")
parser.add_argument("--actor_history", type=int, default=1)
parser.add_argument("--noise_std_floor", type=float, default=None)
parser.add_argument("--motor_curves", type=Path, help="JSON map: motor group -> output-shaft [rad/s, Nm] points")
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--allow_auxiliary_change", action="store_true",
                    help="Explicitly fine-tune matching v2 weights after changing auxiliary joint control")
parser.add_argument("--mass_randomization", action="store_true")
parser.add_argument("--pushes", action="store_true")
parser.add_argument("--log_root", type=Path, default=bootstrap.ROOT / "logs")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.allow_auxiliary_change and (args.resume or not args.checkpoint):
    parser.error("--allow_auxiliary_change requires --checkpoint and cannot be used with --resume")
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
from wheel_legged_lab.runners import NoiseFloorRunner, clamp_policy_noise

def main():
    settings = settings_for(args.task, args.robot_spec, args.strategy)
    settings.observations.actor_history = args.actor_history
    if args.motor_curves:
        settings.motors.torque_speed_curves = json.loads(args.motor_curves.read_text())
    settings.mass.enabled = args.mass_randomization
    settings.force.enabled = args.pushes
    cfg = WheelLeggedEnvCfg(settings=settings)
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, args.seed
    agent = WheelLeggedPPORunnerCfg()
    agent.device, agent.seed, agent.max_iterations = args.device, args.seed, args.max_iterations
    floor = args.noise_std_floor if args.noise_std_floor is not None else (0.1 if args.strategy == "legacy_ppo" else 0.0)
    if not 0 <= floor < float("inf"):
        raise ValueError("noise_std_floor must be finite and nonnegative")
    if args.resume and not args.checkpoint:
        raise ValueError("--resume requires --checkpoint")
    log = args.log_root / args.task / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log.mkdir(parents=True, exist_ok=False)
    env = RslRlVecEnvWrapper(gym.make(f"WheelLegged-{args.task.title()}-Direct-v0", cfg=cfg))
    try:
        schema = env.unwrapped.observation_schema
        save_schema(log, schema)
        dump_yaml(str(log / "env.yaml"), cfg)
        agent_dict = agent.to_dict()
        agent_dict["noise_std_floor"] = floor
        dump_yaml(str(log / "agent.yaml"), agent_dict)
        (log / "launch.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
        runner_type = NoiseFloorRunner if floor > 0 else OnPolicyRunner
        runner = runner_type(env, agent_dict, log_dir=str(log), device=args.device)
        if args.checkpoint:
            validate_checkpoint(args.checkpoint, schema, allow_auxiliary_change=args.allow_auxiliary_change)
            runner.load(str(args.checkpoint), load_optimizer=args.resume, map_location=args.device)
            if args.resume:
                runner.current_learning_iteration += 1
            else:
                runner.current_learning_iteration = 0
                with torch.no_grad():
                    runner.alg.policy.std.fill_(agent.policy.init_noise_std)
        clamp_policy_noise(runner.alg.policy, floor)
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    finally:
        env.close()

try:
    main()
finally:
    app.close()
