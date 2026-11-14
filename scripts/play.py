"""Deterministic playback with optional keyboard commands adapted from the old project."""
import argparse
import json
import sys
import traceback
import time
import math
from datetime import datetime
from itertools import count
from pathlib import Path
import bootstrap
from isaaclab.app import AppLauncher
from playback_config import resolve_playback_args, restore_runtime_settings

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=("stand", "height", "velocity", "jump", "walk", "spin"), default=None)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--allow_auxiliary_change", action="store_true",
                    help="Explicitly evaluate matching v2 weights with changed auxiliary joint control")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1000, help="Maximum control steps; 0 runs until the window closes")
parser.add_argument("--robot_spec", type=str)
parser.add_argument("--strategy", choices=("baseline", "legacy_ppo"), default=None)
parser.add_argument("--actor_history", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--motor_curves", type=Path)
parser.add_argument("--vx", type=float, default=None)
parser.add_argument("--wz", type=float, default=None)
parser.add_argument("--gimbal_heading", type=float, help="Absolute world heading in radians; spin only")
parser.add_argument("--command_height", type=float, help="Robot target height in meters (not the render/window height)")
parser.add_argument("--jump_at", type=float, default=2.0, help="Seconds until one jump request")
parser.add_argument("--keyboard", action="store_true", help="GUI keyboard playback: W/S, A/D, Z/X, L, Q")
parser.add_argument("--keyboard_speed", type=float, default=2.0, help="Held W/S speed in m/s")
parser.add_argument("--keyboard_turn_rate", type=float, default=2.0, help="Held A/D yaw rate in rad/s")
parser.add_argument("--keyboard_height_rate", type=float, default=0.2, help="Held Z/X height change in m/s")
parser.add_argument("--keyboard_min_height", type=float, default=0.20, help="Keyboard minimum target height in meters")
parser.add_argument("--keyboard_max_height", type=float, default=0.38, help="Keyboard maximum target height in meters")
parser.add_argument("--keyboard_pulse_hold", type=float, default=0.2, help="Hold short/character input pulses for seconds; 0 disables compatibility")
parser.add_argument("--keyboard_log", type=Path, help="Input/command JSONL log; defaults to outputs/playback/keyboard_TIMESTAMP.jsonl")
parser.add_argument("--no_camera_follow", action="store_true", help="Keep manual camera control in keyboard mode")
parser.add_argument("--real-time", "--real_time", dest="real_time", action="store_true", help="Pace playback to simulation time; automatic with --keyboard")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
try:
    saved_schema = resolve_playback_args(args)
    if args.num_envs < 1 or args.steps < 0:
        raise ValueError("num_envs must be positive and steps must be nonnegative")
    if args.keyboard and args.headless:
        raise ValueError("--keyboard requires a GUI window; omit --headless")
    if any(not math.isfinite(v) or v <= 0 for v in
           (args.keyboard_speed, args.keyboard_turn_rate, args.keyboard_height_rate)):
        raise ValueError("Keyboard rates must be finite and positive")
    if not math.isfinite(args.keyboard_pulse_hold) or args.keyboard_pulse_hold < 0:
        raise ValueError("keyboard_pulse_hold must be finite and nonnegative")
    if (not all(math.isfinite(v) for v in (args.keyboard_min_height, args.keyboard_max_height))
            or not 0 < args.keyboard_min_height < args.keyboard_max_height):
        raise ValueError("Keyboard height bounds must be finite, positive and increasing")
    if args.keyboard and args.command_height is not None and not args.keyboard_min_height <= args.command_height <= args.keyboard_max_height:
        raise ValueError("command_height must lie inside the keyboard height bounds")
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
    if args.keyboard and not args.no_camera_follow:
        cfg.viewer.origin_type = "asset_root"
        cfg.viewer.asset_name = "robot"
        cfg.viewer.env_index = 0
        cfg.viewer.eye, cfg.viewer.lookat = (3., 3., 2.), (0., 0., 0.)
    env = RslRlVecEnvWrapper(gym.make(f"WheelLegged-{args.task.title()}-Direct-v0", cfg=cfg))
    keyboard = None
    try:
        u = env.unwrapped
        validate_checkpoint(args.checkpoint, u.observation_schema, allow_auxiliary_change=args.allow_auxiliary_change)
        runner = OnPolicyRunner(env, WheelLeggedPPORunnerCfg().to_dict(), log_dir=None, device=args.device)
        runner.load(str(args.checkpoint), load_optimizer=False, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        if args.keyboard:
            u.enable_keyboard_control((args.keyboard_min_height, args.keyboard_max_height))
        print(f"[INFO] Policy loaded. Replaying {'until window closes' if args.steps == 0 else str(args.steps) + ' steps'}.", flush=True)
        with torch.inference_mode():
            obs, _ = env.reset()
            vx = args.vx if args.vx is not None else (0.5 if args.task == "walk" and not args.keyboard else 0.0)
            wz = args.wz if args.wz is not None else (2.0 if args.task == "spin" and not args.keyboard else 0.0)
            u.set_play_velocity_command(vx=vx, vy=0.0, wz=wz)
            if args.gimbal_heading is not None:
                u.set_gimbal_heading(args.gimbal_heading)
            if args.command_height is not None:
                u.set_height_command(args.command_height)
            if args.keyboard:
                if not u.sim.has_gui():
                    raise ValueError("Keyboard playback requires a GUI window")
                from wheel_legged_lab.components.keyboard import KeyboardCommandState, PlaybackKeyboard
                state = KeyboardCommandState(settings, float(u.commands.height[0]),
                    speed=args.keyboard_speed, turn_rate=args.keyboard_turn_rate, height_rate=args.keyboard_height_rate,
                    fixed_vx=args.vx, fixed_wz=args.wz, fixed_height=args.command_height, pulse_hold=args.keyboard_pulse_hold,
                    unified=True, height_range=(args.keyboard_min_height, args.keyboard_max_height))
                log_path = args.keyboard_log or (Path(__file__).resolve().parents[1] / "outputs/playback" /
                    f"keyboard_{datetime.now():%Y-%m-%d_%H-%M-%S_%f}.jsonl")
                keyboard = PlaybackKeyboard(u, state, log_path=log_path)
                print(keyboard.help_text(), flush=True)
                keyboard.advance(0.)
                keyboard.report_jump_status()
            obs = env.get_observations()
            for step in count():
                if not app.is_running() or (args.steps and step >= args.steps):
                    break
                started = time.perf_counter()
                if keyboard is not None:
                    keyboard.advance(u.step_dt)
                    # Replace the current observation frame, preserving configured sensor delay.
                    obs = env.get_observations()
                if args.task == "jump" and not args.keyboard and step == round(args.jump_at / u.step_dt):
                    u.request_jump_takeoff()
                obs, _, dones, _ = env.step(policy(obs))
                if keyboard is not None:
                    keyboard.report_jump_status()
                if keyboard is not None and bool(dones.any()):
                    keyboard.reset()
                    keyboard.advance(0.)
                    print("[INFO] Episode reset: keyboard commands cleared.", flush=True)
                if args.keyboard or args.real_time:
                    time.sleep(max(0., u.step_dt - (time.perf_counter() - started)))
    finally:
        try:
            if keyboard is not None:
                keyboard.close()
        finally:
            env.close()

try:
    main()
except KeyboardInterrupt:
    print("[INFO] Playback interrupted.", flush=True)
except BaseException:
    # Kit fast shutdown can terminate Python before an unwinding exception is
    # displayed. Print and flush the actual failure before closing the app.
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    app.close()
