"""Evaluate measured jumps as independent original-episode trials; never train."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import traceback
import bootstrap
from isaaclab.app import AppLauncher
from playback_config import resolve_playback_args, restore_runtime_settings

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--mode',choices=('smoke','evaluate'),default='evaluate')
parser.add_argument('--transfer_actor',action='store_true',help='Smoke only: seed a fresh jump actor from compatible stand/walk weights')
parser.add_argument('--num_envs',type=int,default=8)
parser.add_argument('--rounds',type=int,default=3)
parser.add_argument('--settle_s',type=float,default=2.5)
parser.add_argument('--trial_s',type=float,default=3.)
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--strategy',choices=('baseline','legacy_ppo'),default=None)
parser.add_argument('--actor_history',type=int,default=None)
parser.add_argument('--robot_spec')
parser.add_argument('--trace',action='store_true')
parser.add_argument('--output_dir',type=Path,default=bootstrap.ROOT/'outputs/jump_evaluation'/datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
try:
    args.task=None
    saved=resolve_playback_args(args)
    if args.transfer_actor and args.mode!='smoke':
        raise ValueError('Actor transfer is an execution smoke test, not a trained-jump evaluation')
    if not args.transfer_actor and (args.task!='jump' or not saved.get('jump_task')):
        raise ValueError('Evaluate requires a measured-jump checkpoint; for a stand/walk seed use --mode smoke --transfer_actor')
    if min(args.num_envs,args.rounds)<1 or any(not math.isfinite(v) or v<=0 for v in (args.settle_s,args.trial_s)):
        raise ValueError('Environment/round counts and trial durations must be positive and finite')
    args.output_dir=args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True,exist_ok=False)
except (ValueError,OSError) as exc:
    parser.error(str(exc))
app=AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import wheel_legged_lab
from wheel_legged_lab.settings import settings_for
from wheel_legged_lab.config import WheelLeggedEnvCfg
from wheel_legged_lab.ppo import WheelLeggedPPORunnerCfg
from wheel_legged_lab.checkpoints import validate_checkpoint,transfer_locomotion_actor


def main():
    settings=settings_for('jump',args.robot_spec,args.strategy)
    settings.observations.actor_history=args.actor_history
    if not args.transfer_actor:
        restore_runtime_settings(settings,saved)
    settings.observations.noise_enabled=False
    settings.jump.automatic=False
    if args.settle_s<settings.jump.min_episode_s or args.trial_s<settings.jump.timeout_s+.5:
        raise ValueError('Settle must cover minimum real episode age; trial must cover timeout plus 0.5 s landing survival')
    cfg=WheelLeggedEnvCfg(settings=settings)
    cfg.scene.num_envs,cfg.sim.device,cfg.seed=args.num_envs,args.device,args.seed
    env=RslRlVecEnvWrapper(gym.make('WheelLegged-Jump-Direct-v0',cfg=cfg))
    trace=(args.output_dir/'trace.jsonl').open('w',buffering=1) if args.trace else None
    try:
        u=env.unwrapped;u.record_jump_transitions=True
        if args.mode=='smoke':
            # Exercise the lifecycle using a standing seed. Only the entry
            # height gate changes; measured takeoff/landing gates stay strict.
            u.jump.ready_height_tolerance=max(settings.jump.height_tolerance,.2)
        runner=OnPolicyRunner(env,WheelLeggedPPORunnerCfg().to_dict(),log_dir=None,device=args.device)
        audit=None
        if args.transfer_actor:
            audit=transfer_locomotion_actor(runner.alg.policy,args.checkpoint,u.observation_schema,device=args.device)
        else:
            validate_checkpoint(args.checkpoint,u.observation_schema)
            runner.load(str(args.checkpoint),load_optimizer=False,map_location=args.device)
        policy=runner.get_inference_policy(device=args.device)
        trials=[];finite=True;land_observation_checks=0
        with torch.inference_mode():
            for round_index in range(args.rounds):
                obs,_=env.reset()
                u.set_play_velocity_command(vx=0.,vy=0.,wz=0.)
                obs=env.get_observations()
                dead=torch.zeros(u.num_envs,dtype=torch.bool,device=u.device)
                flags={name:torch.zeros_like(dead) for name in ('triggered','released','airborne','landed','failure')}
                peaks=torch.zeros(u.num_envs,device=u.device)
                for step in range(round(args.settle_s/u.step_dt)+round(args.trial_s/u.step_dt)):
                    if not app.is_running():
                        raise RuntimeError('Simulation closed before evaluation finished')
                    requested_step=round(args.settle_s/u.step_dt)
                    if step==requested_step:
                        eligible=~dead.clone()
                        u.request_jump_takeoff(env_ids=eligible.nonzero(as_tuple=True)[0])
                        # Multiple observation reads must not advance phase timers.
                        elapsed=u.jump.elapsed.clone();phase=u.jump.phase.clone()
                        obs=env.get_observations();env.get_observations()
                        assert torch.equal(elapsed,u.jump.elapsed) and torch.equal(phase,u.jump.phase)
                    actions=policy(obs)
                    observation_values=(obs,) if torch.is_tensor(obs) else obs.values()
                    finite=(finite and bool(torch.isfinite(actions).all())
                            and all(bool(torch.isfinite(value).all()) for value in observation_values))
                    obs,reward,dones,info=env.step(actions)
                    transition=info['jump_transition']
                    finite=finite and bool(torch.isfinite(reward).all()) and bool(transition['finite'].all())
                    alive=~dead
                    if step>=requested_step:
                        for name,event in (('triggered','trigger_event'),('released','release_event'),
                            ('airborne','takeoff_event'),('landed','success_event'),('failure','failure_event')):
                            flags[name] |= transition[event] & alive
                        peaks=torch.where(alive & flags['triggered'],torch.maximum(peaks,transition['peak']),peaks)
                    landing=transition['phase']==u.jump.LAND
                    if landing.any():
                        control=transition['control_observation'][landing]
                        assert bool((control[:,0]==1).all()) and bool((control[:,1:]==0).all())
                        land_observation_checks+=int(landing.sum())
                    if trace:
                        data={name:value.cpu().tolist() for name,value in transition.items()}
                        trace.write(json.dumps(dict(round=round_index,step=step,**data))+'\n')
                    dead |= dones.bool()
                for index in range(u.num_envs):
                    row=dict(round=round_index,env=index,requested=bool(eligible[index]),reset=bool(dead[index]),
                        survived_success=bool(flags['landed'][index] & ~dead[index]),peak_height_m=float(peaks[index]))
                    row.update({name:bool(value[index]) for name,value in flags.items()})
                    trials.append(row)
        counts={name:sum(row[name] for row in trials) for name in
            ('requested','triggered','released','airborne','landed','survived_success','failure','reset')}
        report=dict(mode=args.mode,checkpoint=str(args.checkpoint),actor_transfer=audit,seed=args.seed,
            num_envs=args.num_envs,rounds=args.rounds,finite=finite,counts=counts,
            takeoff_per_attempt=counts['airborne']/counts['triggered'] if counts['triggered'] else None,
            stable_landing_per_attempt=counts['survived_success']/counts['triggered'] if counts['triggered'] else None,
            untriggered_requests=counts['requested']-counts['triggered'],landing_observation_checks=land_observation_checks,
            settle_s=args.settle_s,trial_s=args.trial_s,trial_records=trials,training_started=False,
            ready_height_tolerance=u.jump.ready_height_tolerance,
            landing_height_tolerance=settings.jump.height_tolerance,
            note='Smoke validates execution only. Success requires measured takeoff, stable landing and survival to trial end; a reset never substitutes a new episode.')
        (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        if not finite:raise RuntimeError('Non-finite state/action/observation/reward detected; see summary.json')
        if args.mode=='smoke' and not counts['triggered']:
            raise RuntimeError('Smoke did not exercise a jump attempt; inspect readiness in trace.jsonl')
    finally:
        if trace:trace.close()
        env.close()

try:
    main()
except BaseException:
    traceback.print_exc();sys.stderr.flush();raise
finally:
    app.close()
