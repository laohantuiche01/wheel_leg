"""Walk/spin command semantics, world heading, reference frames and actor transfer."""
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType('wheel_legged_lab')
package.__path__ = [str(ROOT / 'src/wheel_legged_lab')]
sys.modules.setdefault('wheel_legged_lab', package)
from wheel_legged_lab.settings import settings_for, TASKS
from wheel_legged_lab.robot import RobotDescription
from wheel_legged_lab.components.locomotion import LocomotionCommands, GimbalHeadingController, world_to_heading
from wheel_legged_lab.components.observations import observation_schema, ObservationBuilder
from wheel_legged_lab.components.state import StateReader
from wheel_legged_lab.components.rewards import RewardModel
from wheel_legged_lab.checkpoints import transfer_locomotion_actor, validate_checkpoint, save_schema


def quaternion(yaw):
    yaw = torch.as_tensor(yaw, dtype=torch.float32)
    return torch.stack(((yaw/2).cos(), torch.zeros_like(yaw), torch.zeros_like(yaw), (yaw/2).sin()), dim=-1)


class DummyPolicy(torch.nn.Module):
    def __init__(self, actor_dim, critic_dim):
        super().__init__()
        self.actor = torch.nn.Sequential(torch.nn.Linear(actor_dim, 8), torch.nn.ELU(), torch.nn.Linear(8, 6))
        self.critic = torch.nn.Sequential(torch.nn.Linear(critic_dim, 8), torch.nn.ELU(), torch.nn.Linear(8, 1))
        self.std = torch.nn.Parameter(torch.ones(6) * .25)


class LocomotionTaskTest(unittest.TestCase):
    def setUp(self):
        self.spec = RobotDescription.load(ROOT/'src/wheel_legged_lab/assets/v2_0.json')

    def test_task_settings_and_signed_spin_commands(self):
        for task in TASKS:
            settings_for(task).validate()
        s = settings_for('spin')
        commands = LocomotionCommands(s.commands, 512, 'cpu')
        commands.reset(torch.arange(512))
        requested = commands.sampler.velocity
        self.assertTrue((requested[:, 0:2] == 0).all())
        self.assertTrue((requested[:, 2].abs() >= 1).all())
        self.assertTrue((requested[:, 2].abs() <= 2).all())
        self.assertTrue((requested[:, 2] > 0).any() and (requested[:, 2] < 0).any())
        self.assertTrue((commands.velocity == 0).all())
        walk = settings_for('walk')
        self.assertEqual(walk.commands.spin_fraction, 0)
        self.assertEqual(walk.rewards.velocity_reference, 'axle_heading')

    def test_translation_and_spin_reject_mixed_manual_commands(self):
        ids = torch.arange(256)
        walk = LocomotionCommands(settings_for('walk').commands, len(ids), 'cpu')
        walk.reset(ids)
        for _ in range(500):
            walk.advance(.02)
            self.assertTrue((walk.velocity[:, 1:] == 0).all())
        self.assertTrue((walk.sampler.velocity[:, 0] > 0).any())
        self.assertTrue((walk.sampler.velocity[:, 0] < 0).any())
        with self.assertRaisesRegex(ValueError, 'Translation-only'):
            walk.set_manual(ids, vx=1., wz=.5)
        self.assertTrue(torch.isnan(walk.sampler.manual_velocity).all())
        spin = LocomotionCommands(settings_for('spin').commands, len(ids), 'cpu')
        spin.reset(ids)
        with self.assertRaisesRegex(ValueError, 'In-place spin'):
            spin.set_manual(ids, vx=1., wz=2.)
        self.assertTrue(torch.isnan(spin.sampler.manual_velocity).all())

    def test_command_settle_slew_manual_persistence_and_subset_reset(self):
        s = settings_for('velocity')  # generic ramp supports legacy mixed commands
        c = LocomotionCommands(s.commands, 2, 'cpu')
        ids = torch.arange(2)
        c.reset(ids)
        c.set_manual(ids, vx=1., wz=.6)
        for _ in range(40):
            c.advance(.02)
        torch.testing.assert_close(c.velocity, torch.zeros(2, 3))
        previous = c.velocity.clone()
        for _ in range(200):
            c.advance(.02)
            self.assertTrue(((c.velocity - previous).abs() <= torch.tensor([.016001, .016001, .030001])).all())
            previous = c.velocity.clone()
        torch.testing.assert_close(c.velocity, torch.tensor([[1.,0.,.6], [1.,0.,.6]]))
        c.reset(torch.tensor([0]))
        torch.testing.assert_close(c.velocity[0], torch.zeros(3))
        torch.testing.assert_close(c.velocity[1], torch.tensor([1.,0.,.6]))
        with self.assertRaises(ValueError):
            c.set_manual(ids, vy=1.)

    def fake_heading_controller(self):
        data = types.SimpleNamespace(root_quat_w=quaternion([0., 0.]),
            default_joint_pos=torch.zeros(2, 2), joint_pos=torch.zeros(2, 2), joint_vel=torch.zeros(2, 2),
            body_quat_w=quaternion([[0.,0.], [0.,0.]]), body_ang_vel_w=torch.zeros(2, 2, 3))
        writes = []
        robot = types.SimpleNamespace(data=data, joint_names=['gimbal_yaw_joint','gimbal_pitch_joint'],
            body_names=['base_link','gimbal_yaw_link'],
            set_joint_effort_target=lambda effort, joint_ids: writes.append((effort.clone(), joint_ids)))
        cfg = settings_for('spin').spin
        ctrl = GimbalHeadingController(robot, self.spec, cfg, 2, 'cpu')
        ctrl.reset(torch.arange(2))
        return ctrl, robot, writes

    def test_world_heading_is_not_relative_joint_damping(self):
        ctrl, robot, writes = self.fake_heading_controller()
        # Chassis turns +1 rad while gimbal counter-turns -1 rad: world yaw is 0.
        robot.data.root_quat_w[:] = quaternion([1., 1.])
        robot.data.joint_pos[:, 0] = -1
        robot.data.joint_vel[:, 0] = -2
        ctrl.apply()
        torch.testing.assert_close(writes[-1][0], torch.zeros(2, 1))
        self.assertEqual(writes[-1][1], [0])
        self.assertEqual(tuple(ctrl.observation().shape), (2, 5))
        robot.data.body_quat_w[:, 1] = quaternion([.5, -.5])
        ctrl.apply()
        torch.testing.assert_close(writes[-1][0][:, 0], torch.tensor([-2., 2.]))

    def test_heading_wrap_reset_and_manual_target(self):
        ctrl, robot, _ = self.fake_heading_controller()
        ctrl.set_target(torch.tensor([0]), math.pi-.01)
        robot.data.body_quat_w[0, 1] = quaternion(-math.pi+.01)
        self.assertAlmostEqual(float(ctrl.error()[0]), -.02, places=5)
        robot.data.root_quat_w[1] = quaternion(1.2)
        ctrl.reset(torch.tensor([1]))
        self.assertAlmostEqual(float(ctrl.target[1]), 1.2, places=5)
        ctrl.reset(torch.tensor([0]))
        self.assertAlmostEqual(float(ctrl.target[0]), math.pi-.01, places=5)
        with self.assertRaises(ValueError):
            ctrl.set_target(torch.arange(2), float('nan'))

    def test_spin_actuator_has_single_writer(self):
        def cfg(**kwargs):
            return types.SimpleNamespace(**kwargs)
        class ArticulationCfg:
            InitialStateCfg = staticmethod(cfg)
            def __init__(self, **kwargs):
                vars(self).update(kwargs)
        parent = types.ModuleType('isaaclab')
        parent.__path__ = []
        sim = types.ModuleType('isaaclab.sim')
        sim.UsdFileCfg = sim.RigidBodyPropertiesCfg = sim.ArticulationRootPropertiesCfg = cfg
        modules = {'isaaclab':parent,'isaaclab.sim':sim,
            'isaaclab.assets':types.SimpleNamespace(ArticulationCfg=ArticulationCfg),
            'isaaclab.actuators':types.SimpleNamespace(IdealPDActuatorCfg=cfg)}
        with patch.dict(sys.modules, modules):
            built = self.spec.build_articulation(settings_for('spin').spin)
        owners = [name for name, actuator in built.actuators.items() if 'gimbal_yaw_joint' in actuator.joint_names_expr]
        self.assertEqual(owners, ['gimbal_heading'])
        self.assertEqual(built.actuators['gimbal_heading'].damping, 0.)
        spec = deepcopy(self.spec)
        spec.auxiliary_holds['gimbal_yaw_joint'] = spec.auxiliary_holds['gimbal_pitch_joint']
        with patch.dict(sys.modules, modules), self.assertRaises(ValueError):
            spec.build_articulation(settings_for('spin').spin)

    def test_axle_reference_excludes_cad_base_orbit(self):
        d = types.SimpleNamespace(root_state_w=torch.zeros(1,13), root_pos_w=torch.tensor([[0.,0.,.35]]),
            root_quat_w=quaternion([0.]), root_lin_vel_b=torch.tensor([[0.,.34,0.]]),
            root_lin_vel_w=torch.tensor([[0.,.34,0.]]), root_ang_vel_b=torch.tensor([[0.,0.,2.]]),
            root_ang_vel_w=torch.tensor([[0.,0.,2.]]), projected_gravity_b=torch.tensor([[0.,0.,-1.]]),
            joint_pos=torch.zeros(1,6), joint_vel=torch.zeros(1,6),
            body_link_lin_vel_w=torch.zeros(1,3,3),
            body_pos_w=torch.tensor([[[0.,0.,.35],[-.17,.2,.06],[-.17,-.2,.06]]]))
        sensor = types.SimpleNamespace(body_names=['base','left','right'],data=types.SimpleNamespace(
            net_forces_w=torch.tensor([[[0.,0.,0.],[0.,0.,50.],[0.,0.,50.]]])))
        binding = types.SimpleNamespace(wheel_body_ids=[1,2], wheel_sensor_ids=[1,2], termination_sensor_ids=[0])
        state = StateReader(types.SimpleNamespace(data=d),sensor,binding,self.spec,torch.zeros(1,3),5.).read()
        torch.testing.assert_close(state.axle_linear_velocity_heading, torch.zeros(1,3))
        self.assertAlmostEqual(float(state.linear_velocity[0,1]), .34, places=5)
        torch.testing.assert_close(world_to_heading(torch.tensor([[0.,1.,0.]]), quaternion([math.pi/2])), torch.tensor([[1.,0.,0.]]), atol=1e-6, rtol=1e-6)
        state.angular_velocity[:,2] = .7  # world and body z rates need not be equal
        commands = types.SimpleNamespace(velocity=torch.tensor([[0.,0.,2.]]),height=torch.tensor([.35]))
        motors = types.SimpleNamespace(actions=torch.zeros(1,6),previous=torch.zeros(1,6),effort=torch.zeros(1,6))
        jump = types.SimpleNamespace(IDLE=0,LAND=3,PUSH=1,phase=torch.zeros(1,dtype=torch.long),
            push_reference=lambda:(torch.zeros(1),torch.zeros(1)),air_now=torch.zeros(1,dtype=torch.bool),
            seen_air=torch.zeros(1,dtype=torch.bool),exit_event=torch.zeros(1,dtype=torch.bool),
            success_event=torch.zeros(1,dtype=torch.bool),peak=torch.zeros(1),target_peak=torch.zeros(1))
        reward = RewardModel(types.SimpleNamespace(velocity_reference='axle_heading',weights=dict(spin_drift=-3.,yaw_error=-1.)),1,'cpu')
        torch.testing.assert_close(reward.compute(state,commands,motors,jump,torch.zeros(1,dtype=torch.bool),.02),torch.zeros(1))

    def test_spin_observation_dimensions(self):
        ctrl, robot, _ = self.fake_heading_controller()
        cfg = settings_for('spin').observations
        cfg.noise_enabled, cfg.actor_history = False, 2
        robot.data.joint_pos = torch.zeros(2,6)
        robot.data.default_joint_pos = torch.zeros(2,6)
        robot.data.joint_vel = torch.zeros(2,6)
        motors = types.SimpleNamespace(actions=torch.zeros(2,6),ids=list(range(6)),directions=torch.ones(6),
            position_mask=torch.ones(6),effort=torch.zeros(2,6),strength=torch.ones(2,6))
        builder = ObservationBuilder(cfg,robot,motors,types.SimpleNamespace(velocity=torch.zeros(2,3),height=torch.ones(2)),
            types.SimpleNamespace(observation=lambda:torch.zeros(2,7)),types.SimpleNamespace(scale=torch.ones(2,1)),
            types.SimpleNamespace(force=torch.zeros(2,1,3),torque=torch.zeros(2,1,3)),ctrl)
        state = types.SimpleNamespace(angular_velocity=torch.zeros(2,3),gravity=torch.tensor([[0.,0.,-1.]]).repeat(2,1),
            linear_velocity=torch.zeros(2,3),height=torch.ones(2),wheel_contact=torch.ones(2,2))
        obs = builder.build(state,1)
        self.assertEqual(tuple(obs['policy'].shape),(2,80))
        self.assertEqual(tuple(obs['critic'].shape),(2,65))
        self.assertEqual(observation_schema(self.spec,cfg,settings_for('spin').spin)['actor_dim'],80)

    def schema(self, settings):
        spec = self.spec
        schema = observation_schema(spec,settings.observations,settings.spin)
        schema.update(asset_sha256='same-test-asset',motor_groups=[asdict(g) for g in spec.motors],
            rest_joint_positions=spec.rest_joint_positions,rest_height=spec.rest_height,wheel_radius=spec.wheel_radius,
            spring_settings=asdict(settings.spring),auxiliary_holds={n:asdict(h) for n,h in spec.auxiliary_holds.items()},
            observation_settings=asdict(settings.observations))
        if settings.task in ('walk','spin'):
            schema['motion_task'] = dict(task=settings.task, spin=asdict(settings.spin), commands=asdict(settings.commands))
        return json.loads(json.dumps(schema))

    def test_actor_transfer_preserves_mean_per_history_frame_and_keeps_critic_fresh(self):
        for history in (1,3):
            old_settings, new_settings = settings_for('stand'),settings_for('spin')
            old_settings.observations.actor_history = new_settings.observations.actor_history = history
            old, new = DummyPolicy(35*history,60),DummyPolicy(40*history,65)
            critic = {k:v.clone() for k,v in new.critic.state_dict().items()}
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'model.pt'
                torch.save(dict(model_state_dict=old.state_dict()),path)
                save_schema(directory,self.schema(old_settings))
                target = self.schema(new_settings)
                with self.assertRaises(ValueError):
                    validate_checkpoint(path,target)
                transfer_locomotion_actor(new,path,target)
                x = torch.randn(4,history,35)
                expanded = torch.randn(4,history,40)
                expanded[:,:,:35] = x
                torch.testing.assert_close(old.actor(x.flatten(1)),new.actor(expanded.flatten(1)))
                for key,value in new.critic.state_dict().items():
                    torch.testing.assert_close(value,critic[key])
                target['action_directions'][-1] *= -1
                with self.assertRaises(ValueError):
                    transfer_locomotion_actor(new,path,target)

    def test_existing_spin_actor_can_seed_new_spin_command_range(self):
        old_settings, new_settings = settings_for('spin'), settings_for('spin')
        old_settings.commands.wz = (-3., 3.)
        old, new = DummyPolicy(40,65), DummyPolicy(40,65)
        critic = {k:v.clone() for k,v in new.critic.state_dict().items()}
        with torch.no_grad():
            old.std.fill_(.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            torch.save(dict(model_state_dict=old.state_dict()),path)
            save_schema(directory,self.schema(old_settings))
            target = self.schema(new_settings)
            with self.assertRaises(ValueError):
                validate_checkpoint(path,target)
            audit = transfer_locomotion_actor(new,path,target)
            self.assertEqual(audit['zeroed_new_columns'],0)
            x = torch.randn(4,40)
            torch.testing.assert_close(new.actor(x),old.actor(x))
            torch.testing.assert_close(new.std,torch.full((6,),.25))
            for key,value in new.critic.state_dict().items():
                torch.testing.assert_close(value,critic[key])
            target['motion_task']['spin']['kp'] *= 2
            with self.assertRaisesRegex(ValueError, 'gimbal control'):
                transfer_locomotion_actor(new,path,target)


if __name__ == '__main__':
    unittest.main()
