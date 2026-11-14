"""Measured release, true flight, standing recovery and bounded jump rewards."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
package=types.ModuleType('wheel_legged_lab');package.__path__=[str(ROOT/'src/wheel_legged_lab')]
sys.modules.setdefault('wheel_legged_lab',package)
sys.path.insert(0,str(ROOT/'scripts'))
from wheel_legged_lab.settings import settings_for, apply_jump_reward_profile
from wheel_legged_lab.components.jump_task import MeasuredJumpController, jump_task_schema
from wheel_legged_lab.components.rewards import RewardModel
from wheel_legged_lab.checkpoints import save_schema,validate_checkpoint,transfer_locomotion_actor
from playback_config import restore_runtime_settings


class JumpTaskTest(unittest.TestCase):
    def make(self,n=1,min_age=0,profile='measured_v1'):
        s=settings_for('jump');s.jump.automatic=False;s.jump.min_episode_s=min_age
        s.jump.implementation=profile;apply_jump_reward_profile(s)
        j=MeasuredJumpController(s.jump,n,'cpu')
        state=types.SimpleNamespace(height=torch.full((n,),.28),vertical_velocity=torch.zeros(n),
            wheel_clearance=torch.zeros(n,2),wheel_contact=torch.ones(n,2,dtype=torch.bool),
            wheel_contact_force=torch.full((n,2),50.),finite=torch.ones(n,dtype=torch.bool),
            forbidden_contact=torch.zeros(n,dtype=torch.bool),non_wheel_contact=torch.zeros(n,dtype=torch.bool),
            gravity=torch.tensor([0.,0.,-1.]).repeat(n,1),angular_velocity=torch.zeros(n,3),linear_velocity=torch.zeros(n,3),
            wheel_fore_aft_difference=torch.zeros(n))
        commands=types.SimpleNamespace(height=torch.full((n,),.28),velocity=torch.zeros(n,3))
        return s,j,state,commands

    def step(self,j,state,commands,count=1):
        for _ in range(count):j.advance(.02,state,commands,torch.full_like(state.height,999.))

    def enter(self,j,state,commands):
        j.request(torch.arange(len(j.phase)));self.step(j,state,commands)
        self.assertTrue((j.phase==j.PUSH).all())

    def flight(self,j,state,commands):
        state.vertical_velocity[:]=1.4;state.height[:]=.34
        self.step(j,state,commands)
        self.assertTrue(j.release_event.all())
        state.height[:]=.4;state.wheel_clearance[:]=.03;state.wheel_contact[:]=False;state.wheel_contact_force[:]=0
        self.step(j,state,commands,2)
        self.assertTrue(j.takeoff_event.all())

    def landing(self,j,state,commands,count=1):
        state.height[:]=.28;state.vertical_velocity[:]=0
        state.wheel_clearance[:]=0;state.wheel_contact[:]=True;state.wheel_contact_force[:]=50
        self.step(j,state,commands,count)

    def test_settings_are_in_place_and_parameters_are_validated(self):
        s=settings_for('jump')
        self.assertEqual(s.jump.implementation,'measured_v2')
        self.assertEqual(s.commands.vx,(0.,0.));self.assertEqual(s.commands.wz,(0.,0.))
        for name,value in [('release_speed_fraction',1.1),('min_air_s',0),('push_timeout_s',2),('peak_height',(.2,.3)),('rate_per_s',float('nan'))]:
            other=deepcopy(s);setattr(other.jump,name,value)
            with self.assertRaises(ValueError):other.validate()

    def test_randomized_episode_age_cannot_bypass_actual_settling(self):
        _,j,state,c=self.make(min_age=2.)
        j.request(torch.tensor([0]));self.step(j,state,c,99)
        self.assertEqual(int(j.phase[0]),j.IDLE)
        self.step(j,state,c)
        self.assertEqual(int(j.phase[0]),j.PUSH)

    def test_smoke_entry_override_does_not_relax_landing(self):
        _,j,state,c=self.make();j.ready_height_tolerance=.2;state.height[:]=.36
        self.enter(j,state,c);self.flight(j,state,c);self.landing(j,state,c)
        state.height[:]=.36
        self.step(j,state,c,10)
        self.assertEqual(j.cfg.height_tolerance,.06)
        self.assertFalse(j.success_event.any());self.assertEqual(float(j.stable_time[0]),0.)

    def test_reference_timer_and_height_without_speed_do_not_release(self):
        _,j,state,c=self.make();self.enter(j,state,c)
        state.height[:]=.36
        self.step(j,state,c,13)
        self.assertEqual(int(j.phase[0]),j.PUSH)
        self.assertFalse(j.release_event.any())
        self.step(j,state,c)
        self.assertTrue(j.push_timeout_event.all());self.assertTrue(j.failure_event.all())
        self.assertEqual(float(j.failures[0]),1.)
        self.step(j,state,c,5)
        self.assertFalse(j.failure_event.any());self.assertEqual(float(j.failures[0]),1.)

    def test_release_uses_measured_speed_and_height_or_unloading(self):
        _,j,state,c=self.make();self.enter(j,state,c)
        state.height[:]=.34;state.vertical_velocity[:]=.4
        self.step(j,state,c);self.assertEqual(int(j.phase[0]),j.PUSH)
        state.height[:]=.30;state.vertical_velocity[:]=1.4;state.wheel_contact_force[:]=5
        self.step(j,state,c)
        self.assertEqual(int(j.phase[0]),j.FLIGHT);self.assertTrue(j.release_event.all())
        self.assertEqual(float(j.releases[0]),1.)

    def test_free_fall_and_short_clearance_do_not_count_takeoff(self):
        _,j,state,c=self.make();self.enter(j,state,c)
        state.wheel_contact[:]=False;state.wheel_contact_force[:]=0;state.wheel_clearance[:]=.04
        state.vertical_velocity[:]=-.5
        self.step(j,state,c,3)
        self.assertFalse(j.seen_air.any());self.assertEqual(float(j.takeoffs[0]),0.)
        state.vertical_velocity[:]=.4
        state.wheel_clearance[:]=0
        self.step(j,state,c)
        state.wheel_clearance[:]=.03
        self.step(j,state,c)
        self.assertFalse(j.seen_air.any())
        state.wheel_clearance[:]=0
        self.step(j,state,c)
        self.assertFalse(j.seen_air.any())

    def test_landing_standing_observation_precedes_success_and_is_consecutive(self):
        _,j,state,c=self.make();self.enter(j,state,c);self.flight(j,state,c)
        self.landing(j,state,c,5)
        self.assertEqual(int(j.phase[0]),j.LAND)
        torch.testing.assert_close(j.observation(),torch.tensor([[1.,0.,0.,0.,0.,0.,0.]]))
        self.assertFalse(j.success_event.any())
        state.angular_velocity[:,0]=2.
        self.step(j,state,c)
        self.assertEqual(float(j.stable_time[0]),0.)
        state.angular_velocity[:]=0
        self.landing(j,state,c,10)
        self.assertTrue(j.success_event.all());self.assertEqual(float(j.successes[0]),1.)
        self.assertEqual(int(j.phase[0]),j.IDLE)
        self.step(j,state,c)
        self.assertFalse(j.success_event.any());self.assertEqual(float(j.successes[0]),1.)

    def test_landing_body_contact_blocks_success_and_is_logged(self):
        _,j,state,c=self.make();self.enter(j,state,c);self.flight(j,state,c)
        state.non_wheel_contact[:]=True
        self.landing(j,state,c,12)
        self.assertFalse(j.success_event.any());self.assertEqual(float(j.stable_time[0]),0.)
        state.height[:]=.4;state.vertical_velocity[:]=.5
        self.step(j,state,c)
        logs=j.episode_logs(torch.tensor([0]))
        self.assertGreater(float(logs['Episode/Jump/LandingHeightBlockedFraction']),0.)
        self.assertGreater(float(logs['Episode/Jump/LandingVelZBlockedFraction']),0.)

    def test_active_request_does_not_queue_another_attempt(self):
        _,j,state,c=self.make();j.cfg.cooldown_s=.04;self.enter(j,state,c)
        j.request(torch.tensor([0]));self.step(j,state,c,14)
        self.step(j,state,c,10)
        self.assertEqual(float(j.attempts[0]),1.);self.assertFalse(j.requested.any())

    def test_readonly_diagnostics_identify_waiting_gates_and_push_failure(self):
        _,j,state,c=self.make(min_age=.04)
        j.request(torch.tensor([0]));state.height[:]=.4
        self.step(j,state,c)
        before=j.elapsed.clone()
        status=j.diagnostics()
        self.assertTrue(status['pending'])
        self.assertIn('height',status['blocked']);self.assertIn('minimum_age',status['blocked'])
        torch.testing.assert_close(j.elapsed,before)
        state.height[:]=.28
        self.step(j,state,c)
        self.assertEqual(j.diagnostics()['phase'],'PUSH')
        self.step(j,state,c,14)
        status=j.diagnostics()
        self.assertEqual(status['failures'],1)
        self.assertEqual(status['max_upward_speed'],0.)
        self.assertGreaterEqual(status['release_speed_required'],.3)

    def test_fall_on_landing_tick_cancels_success_and_snapshot_survives_reset(self):
        _,j,state,c=self.make();self.enter(j,state,c);self.flight(j,state,c);self.landing(j,state,c,10)
        self.assertTrue(j.success_event.all())
        j.abort(torch.tensor([True]))
        self.assertFalse(j.success_event.any());self.assertTrue(j.failure_event.all())
        self.assertEqual(float(j.successes[0]),0.);self.assertEqual(float(j.failures[0]),1.)
        saved=j.transition_snapshot(state);j.reset(torch.tensor([0]))
        self.assertTrue(saved['failure_event'].all());self.assertEqual(float(j.attempts[0]),0.)

    def test_push_progress_is_bounded_and_cannot_farm_standing_reward(self):
        s,j,state,c=self.make();self.enter(j,state,c)
        s.rewards.weights={name:0. for name in s.rewards.weights};s.rewards.weights['jump_push_velocity']=1.
        reward=RewardModel(s.rewards,1,'cpu')
        m=types.SimpleNamespace(actions=torch.zeros(1,6),previous=torch.zeros(1,6),effort=torch.zeros(1,6))
        total=0.
        for speed in [0.,0.,.2,.5,.5,.4,.8,1.5,1.5]:
            state.vertical_velocity[:]=speed
            self.step(j,state,c)
            value=float(reward.compute(state,c,m,j,torch.tensor([False]),.02)[0])
            if speed<=.3:self.assertEqual(value,0.)
            total+=value
        self.assertAlmostEqual(total,1.,places=6)

    def test_v2_rewards_low_speed_improvement_without_changing_takeoff_gate(self):
        _,j,state,c=self.make(profile='measured_v2');self.enter(j,state,c)
        state.vertical_velocity[:]=.1
        self.step(j,state,c)
        self.assertGreater(float(j.push_progress_reward[0]),0.)
        self.assertEqual(j.cfg.min_upward_speed,.3)
        self.assertFalse(j.takeoff_event.any());self.assertFalse(j.release_event.any())
        self.step(j,state,c)
        self.assertEqual(float(j.push_progress_reward[0]),0.)
        state.vertical_velocity[:]=0
        self.step(j,state,c)
        state.vertical_velocity[:]=.1
        self.step(j,state,c)
        self.assertEqual(float(j.push_progress_reward[0]),0.)

    def test_v2_height_progress_is_measured_bounded_and_not_repeated(self):
        _,j,state,c=self.make(profile='measured_v2');self.enter(j,state,c)
        total=0.
        for height in [.28,.29,.29,.285,.32,.34,.36]:
            state.height[:]=height
            self.step(j,state,c)
            total+=float(j.height_progress_reward[0])*.02
        self.assertAlmostEqual(total,1.,places=5)
        self.step(j,state,c)
        self.assertEqual(float(j.height_progress_reward[0]),0.)

    def test_v2_phase_observations_disambiguate_push_and_flight(self):
        _,j,state,c=self.make(profile='measured_v2');self.enter(j,state,c)
        torch.testing.assert_close(j.observation()[0,:5],torch.tensor([0.,1.,0.,0.,1.]))
        self.flight(j,state,c)
        torch.testing.assert_close(j.observation()[0,:5],torch.tensor([0.,0.,1.,0.,1.]))
        self.landing(j,state,c)
        torch.testing.assert_close(j.observation()[0],torch.tensor([1.,0.,0.,0.,0.,0.,0.]))

    def test_v1_checkpoint_restores_original_sparse_rewards_and_mode(self):
        s,_,_,_=self.make(profile='measured_v1')
        saved={'jump_task':jump_task_schema(s)}
        self.assertEqual(saved['jump_task']['version'],'measured-jump-v1')
        restored=settings_for('jump');restore_runtime_settings(restored,saved)
        self.assertEqual(restored.jump.implementation,'measured_v1')
        self.assertEqual(restored.rewards.weights['jump_push_velocity'],1.)
        self.assertEqual(restored.rewards.weights['jump_push_height_progress'],0.)
        self.assertEqual(restored.rewards.push_regularization_scale,1.)
        current=settings_for('jump')
        self.assertEqual(jump_task_schema(current)['version'],'measured-jump-v2')
        with tempfile.TemporaryDirectory() as directory:
            save_schema(directory,saved)
            with self.assertRaises(ValueError):validate_checkpoint(Path(directory)/'x.pt',{'jump_task':jump_task_schema(current)})

    def test_v2_push_reduces_effort_cost_while_landing_restores_it(self):
        s,j,state,c=self.make(profile='measured_v2');self.enter(j,state,c)
        s.rewards.weights={name:0. for name in s.rewards.weights};s.rewards.weights['torque']=-1.
        r=RewardModel(s.rewards,1,'cpu')
        m=types.SimpleNamespace(actions=torch.zeros(1,6),previous=torch.zeros(1,6),effort=torch.ones(1,6))
        push=float(r.compute(state,c,m,j,torch.tensor([False]),.02)[0])
        j.phase[:]=j.LAND
        land=float(r.compute(state,c,m,j,torch.tensor([False]),.02)[0])
        self.assertAlmostEqual(push,land*.2,places=6)

    def test_jump_metadata_restores_profile_and_rejects_equal_width_old_weights(self):
        s=settings_for('jump');s.jump.peak_height=(.42,.46)
        schema={'actor_dim':35,'critic_dim':60,'jump_task':jump_task_schema(s)}
        restored=settings_for('jump');restore_runtime_settings(restored,schema)
        self.assertEqual(restored.jump.peak_height,(.42,.46))
        self.assertEqual(jump_task_schema(restored),schema['jump_task'])
        restored.jump.automatic=False
        self.assertEqual(jump_task_schema(restored),schema['jump_task'])
        with tempfile.TemporaryDirectory() as directory:
            save_schema(directory,schema)
            with self.assertRaises(ValueError):validate_checkpoint(Path(directory)/'x.pt',{'actor_dim':35,'critic_dim':60})
        legacy=settings_for('jump');restore_runtime_settings(legacy,{})
        self.assertEqual(legacy.jump.implementation,'legacy')
        self.assertEqual(legacy.rewards.weights['jump_takeoff'],0.)

    def test_explicit_stand_actor_transfer_keeps_critic_fresh(self):
        from test_locomotion_tasks import DummyPolicy
        from wheel_legged_lab.robot import RobotDescription
        from wheel_legged_lab.components.observations import observation_schema
        from wheel_legged_lab.settings import MotorSettings
        spec=RobotDescription.load(ROOT/'src/wheel_legged_lab/assets/v2_0.json')
        schema=observation_schema(spec)
        schema.update(asset_sha256='asset',motor_groups=[asdict(g) for g in spec.motors],
            rest_joint_positions=spec.rest_joint_positions,rest_height=spec.rest_height,wheel_radius=spec.wheel_radius,
            spring_settings=asdict(settings_for('stand').spring),motor_settings=asdict(MotorSettings()))
        source=DummyPolicy(35,60);target=DummyPolicy(35,60)
        critic={k:v.clone() for k,v in target.state_dict().items() if k.startswith('critic.')}
        with tempfile.TemporaryDirectory() as directory:
            save_schema(directory,schema);path=Path(directory)/'model.pt'
            torch.save({'model_state_dict':source.state_dict()},path)
            destination=deepcopy(schema);destination['jump_task']=jump_task_schema(settings_for('jump'))
            audit=transfer_locomotion_actor(target,path,destination)
            self.assertEqual(audit['zeroed_new_columns'],0)
            for key,value in target.state_dict().items():
                if key.startswith('actor.'):
                    expected=source.state_dict()[key].clone()
                    if key=='actor.0.weight':expected[:,[29,30]]=0
                    torch.testing.assert_close(value,expected)
                if key.startswith('critic.'):torch.testing.assert_close(value,critic[key])

if __name__=='__main__':unittest.main()
