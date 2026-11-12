"""Behavior tests for ordinary-PPO mechanisms ported from the old project."""
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("wheel_legged_lab")
package.__path__ = [str(ROOT / "src/wheel_legged_lab")]
sys.modules.setdefault("wheel_legged_lab", package)

from wheel_legged_lab.settings import MotorSettings, TerminationSettings, settings_for, ObservationSettings
from wheel_legged_lab.robot import RobotDescription
from wheel_legged_lab.components.legacy_control import DifferenceVelocity, TorqueSpeedCurve, SustainedTermination, ObservationHistory
from wheel_legged_lab.components.motors import MotorBank
from wheel_legged_lab.components.observations import observation_schema, ObservationBuilder
from wheel_legged_lab.checkpoints import save_schema, validate_checkpoint
from wheel_legged_lab.runners import clamp_policy_noise


class LegacyStrategyTest(unittest.TestCase):
    def test_velocity_wrap_filter_reset_and_first_sample(self):
        v = DifferenceVelocity(2, 2, "cpu", alpha=.5)
        q = torch.tensor([[0., math.pi - .01], [0., 0.]])
        simulator = torch.ones_like(q)
        mask = torch.tensor([False, True])
        torch.testing.assert_close(v.sample(q, simulator, .01, mask), simulator)
        q[0] = torch.tensor([.02, -math.pi + .01])
        result = v.sample(q, simulator, .01, mask).clone()
        torch.testing.assert_close(result[0], torch.tensor([1.5, 1.5]), atol=2e-5, rtol=1e-5)
        v.reset(torch.tensor([0]))
        q[0] = torch.tensor([2., 0.])
        torch.testing.assert_close(v.sample(q, simulator * 3, .01, mask)[0], torch.full((2,), 3.))
        self.assertTrue(v.valid[1])

    def test_measured_curve_interpolation_and_csv_units(self):
        curve = TorqueSpeedCurve([[0, 5], [10, 3], [20, 0]], "cpu")
        torch.testing.assert_close(curve.limit(torch.tensor([-5., 0., 10., 15., 21.])), torch.tensor([4., 5., 3., 1.5, 0.]))
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / 'curve.csv'
            csv.write_text('Motor_shaft_speed_rpm,Motor_shaft_torque_Nm_ideal\n0,1\n600,0.5\n600,0.4\n')
            points = TorqueSpeedCurve.read_csv(csv, gear_ratio=2, efficiency=.8)
            self.assertAlmostEqual(points[1][0], 10 * math.pi)
            self.assertAlmostEqual(points[1][1], .64)
        for points in ([[0, 1], [0, 2]], [[0, 1], [2, float('nan')]], [[0, -1], [1, 0]]):
            with self.assertRaises(ValueError):
                TorqueSpeedCurve(points, 'cpu')

    def test_termination_duration_interrupts_and_immediate_override(self):
        tracker = SustainedTermination(TerminationSettings(duration_steps=3), 2, 'cpu')
        def update(ordinary, immediate, tick):
            return tracker.update(torch.tensor(ordinary), torch.tensor(immediate), tick).clone()
        self.assertFalse(update([True, True], [False, False], 1).any())
        # Repeated observation/done reads cannot advance the persistence timer.
        self.assertFalse(update([True, True], [False, False], 1).any())
        self.assertTrue(update([True, False], [False, True], 2)[1])
        self.assertTrue(update([True, False], [False, False], 3)[0])
        tracker.reset(torch.tensor([0]))
        self.assertFalse(update([False, False], [False, False], 4).any())

    def test_history_order_repeated_reads_and_subset_reset(self):
        h = ObservationHistory(2, 3, 1, 'cpu')
        h.update(torch.tensor([[1.], [10.]]), 1)
        h.update(torch.tensor([[2.], [20.]]), 2)
        actual = h.update(torch.tensor([[3.], [30.]]), 2)
        torch.testing.assert_close(actual, torch.tensor([[0., 1., 3.], [0., 10., 30.]]))
        h.reset(torch.tensor([0]))
        actual = h.update(torch.tensor([[4.], [40.]]), 3)
        torch.testing.assert_close(actual, torch.tensor([[0., 0., 4.], [10., 30., 40.]]))

    def test_action_filter_and_physics_delay(self):
        spec = RobotDescription.load(ROOT / 'src/wheel_legged_lab/assets/v2_0.json')
        data = types.SimpleNamespace(joint_pos=torch.zeros(2, 6), joint_vel=torch.zeros(2, 6),
            default_joint_pos=torch.zeros(2, 6), soft_joint_pos_limits=torch.tensor([-2., 2.]).repeat(2, 6, 1))
        robot = types.SimpleNamespace(data=data, set_joint_effort_target=lambda *args, **kwargs: None)
        cfg = MotorSettings(action_filter_alpha=.8, delay_clock='physics', delay_steps=(2, 2))
        bank = MotorBank(robot, types.SimpleNamespace(action_ids=list(range(6))), spec, cfg, 2, 'cpu')
        bank.reset(torch.tensor([0, 1]))
        bank.accept(torch.ones(2, 6))
        bank.apply()
        torch.testing.assert_close(bank.applied, torch.zeros(2, 6))
        bank.apply()
        torch.testing.assert_close(bank.applied, torch.zeros(2, 6))
        bank.apply()
        torch.testing.assert_close(bank.applied, torch.full((2, 6), .8))
        bank.accept(torch.ones(2, 6))
        torch.testing.assert_close(bank.actions, torch.full((2, 6), .96))
        bank.reset(torch.tensor([0]))
        torch.testing.assert_close(bank.history[:, 0], torch.zeros_like(bank.history[:, 0]))
        self.assertGreater(float(bank.history[:, 1].sum()), 0.)

    def test_noise_floor_for_scalar_and_log_parameterizations(self):
        policy = types.SimpleNamespace(std=torch.nn.Parameter(torch.tensor([.01, .3])),
            log_std=torch.nn.Parameter(torch.log(torch.tensor([.02, .4]))))
        clamp_policy_noise(policy, .1)
        torch.testing.assert_close(policy.std, torch.tensor([.1, .3]))
        torch.testing.assert_close(policy.log_std.exp(), torch.tensor([.1, .4]))

    def test_observation_delay_history_and_checkpoint_boundary(self):
        spec = RobotDescription.load(ROOT / 'src/wheel_legged_lab/assets/v2_0.json')
        cfg = ObservationSettings(noise_enabled=False, actor_history=2, actor_delay=1)
        data = types.SimpleNamespace(joint_pos=torch.zeros(1, 6), default_joint_pos=torch.zeros(1, 6), joint_vel=torch.zeros(1, 6))
        motors = types.SimpleNamespace(actions=torch.ones(1, 6), ids=list(range(6)), position_mask=torch.ones(6),
            directions=torch.ones(6), effort=torch.zeros(1, 6), strength=torch.ones(1, 6))
        commands = types.SimpleNamespace(velocity=torch.zeros(1, 3), height=torch.ones(1))
        builder = ObservationBuilder(cfg, types.SimpleNamespace(data=data), motors, commands,
            types.SimpleNamespace(observation=lambda:torch.zeros(1, 7)), types.SimpleNamespace(scale=torch.ones(1, 1)),
            types.SimpleNamespace(force=torch.zeros(1, 1, 3), torque=torch.zeros(1, 1, 3)))
        state = types.SimpleNamespace(angular_velocity=torch.zeros(1, 3), gravity=torch.tensor([[0., 0., -1.]]),
            linear_velocity=torch.zeros(1, 3), height=torch.ones(1), wheel_contact=torch.ones(1, 2))
        obs = builder.build(state, 1)
        self.assertEqual(tuple(obs['policy'].shape), (1, 70))
        self.assertEqual(tuple(obs['critic'].shape), (1, 60))
        torch.testing.assert_close(obs['policy'], torch.zeros(1, 70))
        obs = builder.build(state, 2)
        self.assertGreater(float(obs['policy'].abs().sum()), 0.)
        with tempfile.TemporaryDirectory() as directory:
            save_schema(directory, observation_schema(spec))
            with self.assertRaises(ValueError):
                validate_checkpoint(Path(directory)/'model.pt', observation_schema(spec, cfg))

    def test_strategy_preserves_stand_model_height_and_posture_rewards(self):
        s = settings_for('stand', strategy='legacy_ppo')
        self.assertEqual(s.motors.delay_clock, 'physics')
        self.assertEqual(s.motors.delay_steps, (1, 3))
        self.assertEqual(s.observations.actor_delay, 1)
        self.assertEqual(s.commands.height, (.372790959,) * 2)
        self.assertEqual(s.rewards.weights['roll_error'], -20.)
        for task in ('height','velocity','jump'):
            settings_for(task, strategy='legacy_ppo').validate()


if __name__ == '__main__':
    unittest.main()
