"""Record walk/spin tracking, axle drift and world-frame gimbal heading."""
import argparse
import csv
import json
from pathlib import Path
import sys
import traceback
import bootstrap
from isaaclab.app import AppLauncher
from playback_config import resolve_playback_args, restore_runtime_settings

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--task', choices=('walk','spin'), default=None)
parser.add_argument('--strategy', choices=('baseline','legacy_ppo'), default=None)
parser.add_argument('--actor_history', type=int, default=None)
parser.add_argument('--robot_spec', type=str)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--num_envs', type=int, default=4)
parser.add_argument('--steps', type=int, default=500)
parser.add_argument('--vx', type=float, default=None)
parser.add_argument('--wz', type=float, default=None)
parser.add_argument('--gimbal_heading', type=float, default=None)
parser.add_argument('--output_dir', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
try:
    saved_schema = resolve_playback_args(args)
    if args.task not in ('walk','spin') or min(args.num_envs,args.steps) < 1:
        raise ValueError('Use a walk/spin checkpoint and positive environment/step counts')
    args.output_dir.mkdir(parents=True, exist_ok=False)
except (ValueError,OSError) as exc:
    parser.error(str(exc))
app = AppLauncher(args).app

import gymnasium as gym
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
    settings = settings_for(args.task,args.robot_spec,args.strategy)
    restore_runtime_settings(settings,saved_schema)
    cfg = WheelLeggedEnvCfg(settings=settings)
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs,args.device,args.seed
    env = RslRlVecEnvWrapper(gym.make(f'WheelLegged-{args.task.title()}-Direct-v0',cfg=cfg))
    try:
        u = env.unwrapped
        validate_checkpoint(args.checkpoint,u.observation_schema)
        runner = OnPolicyRunner(env,WheelLeggedPPORunnerCfg().to_dict(),log_dir=None,device=args.device)
        runner.load(str(args.checkpoint),load_optimizer=False,map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        vx = args.vx if args.vx is not None else (.5 if args.task == 'walk' else 0.)
        wz = args.wz if args.wz is not None else (2. if args.task == 'spin' else 0.)
        names = ['episode_age_s','command_vx','command_wz','axle_vx','axle_vy','world_yaw_rate',
                 'vx_error','yaw_rate_error','height_m','roll_deg','pitch_deg','both_wheels_contact',
                 'non_wheel_contact','gimbal_heading_error_deg','gimbal_world_yaw_rate',
                 'gimbal_relative_yaw_rate','axle_displacement_m']
        rows, resets, completed = [], 0, 0
        with torch.inference_mode(), (args.output_dir/'trajectory.csv').open('w',newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['step','env',*names])
            obs,_ = env.reset()
            u.set_play_velocity_command(vx=vx,vy=0.,wz=wz)
            if args.gimbal_heading is not None:
                u.set_gimbal_heading(args.gimbal_heading)
            obs = env.get_observations()
            anchor = u.reader.read().axle_position_world[:,:2].clone()
            for step in range(args.steps):
                if not app.is_running():
                    break
                command = u.commands.velocity.clone()
                obs,_,dones,_ = env.step(policy(obs))
                resets += int(dones.sum())
                completed = step + 1
                state = u.reader.read()
                anchor[dones.bool()] = state.axle_position_world[dones.bool(),:2]
                roll,pitch = body_tilt_angles(state.gravity)
                zero = torch.zeros(u.num_envs,device=u.device)
                error = state.gimbal_heading_error if u.gimbal_heading is not None else zero
                rate = state.gimbal_world_yaw_rate if u.gimbal_heading is not None else zero
                relative_rate = (u.robot.data.joint_vel[:,u.gimbal_heading.joint_id]
                                 if u.gimbal_heading is not None else zero)
                values = torch.stack((u.episode_length_buf*u.step_dt, command[:,0],command[:,2],
                    state.axle_linear_velocity_heading[:,0],state.axle_linear_velocity_heading[:,1],state.yaw_rate_world,
                    command[:,0]-state.axle_linear_velocity_heading[:,0],command[:,2]-state.yaw_rate_world,
                    state.height,roll.rad2deg(),pitch.rad2deg(),state.wheel_contact.all(dim=1).float(),
                    state.non_wheel_contact.float(),error.rad2deg(),rate,relative_rate,
                    (state.axle_position_world[:,:2]-anchor).norm(dim=1)),dim=1)
                keep = (u.episode_length_buf*u.step_dt >= 2) & ~dones.bool() & state.finite & torch.isfinite(values).all(dim=1)
                rows.append(values[keep].cpu())
                for index,row in enumerate(values.cpu().tolist()):
                    writer.writerow([step,index,*row])
        samples = torch.cat(rows) if rows else torch.empty(0,len(names))
        metrics = {name:dict(mean=float(samples[:,i].mean()),rms=float(samples[:,i].square().mean().sqrt()),
                            min=float(samples[:,i].min()),max=float(samples[:,i].max()))
                   for i,name in enumerate(names)} if len(samples) else {}
        report = dict(task=args.task,checkpoint=str(args.checkpoint),strategy=args.strategy,seed=args.seed,
            num_envs=args.num_envs,requested_steps=args.steps,completed_steps=completed,resets=resets,
            steady_samples=len(samples),manual_command=dict(vx=vx,wz=wz),metrics=metrics,
            note='Execution and tracking measurements; no convergence or robustness guarantee is inferred.',training_started=False)
        (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
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
