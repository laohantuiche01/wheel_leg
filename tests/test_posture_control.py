"""Posture penalties, telemetry and auxiliary-control migration on CPU."""
from copy import deepcopy
from dataclasses import asdict
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

from wheel_legged_lab.robot import RobotDescription
from wheel_legged_lab.components.motors import AuxiliaryHoldController
from wheel_legged_lab.components.rewards import RewardModel
from wheel_legged_lab.components.observations import observation_schema
from wheel_legged_lab.checkpoints import save_schema, validate_checkpoint


class PostureControlTest(unittest.TestCase):
    def test_pitch_hold_restores_position_with_limits_and_damping(self):
        spec = RobotDescription.load(ROOT / "src/wheel_legged_lab/assets/v2_0.json")
        data = types.SimpleNamespace(joint_pos=torch.zeros(2, 30), joint_vel=torch.zeros(2, 30),
            soft_joint_pos_limits=torch.tensor([-.5, .5]).repeat(2, 30, 1))
        written = torch.full((2, 30), float("nan"))
        def write(effort, joint_ids):
            written[:, joint_ids] = effort
        robot = types.SimpleNamespace(data=data, set_joint_effort_target=write)
        hold = AuxiliaryHoldController(robot, types.SimpleNamespace(hold_ids=[7]), spec, 2, "cpu")
        data.joint_pos[:, 7] = torch.tensor([-.5, .5])
        hold.apply()
        torch.testing.assert_close(written[:, 7], torch.tensor([8., -8.]))
        self.assertTrue(torch.isnan(written[:, :7]).all())
        self.assertTrue(torch.isnan(written[:, 8:]).all())
        data.joint_pos[:, 7] = 0
        data.joint_vel[:, 7] = .2
        hold.apply()
        torch.testing.assert_close(written[:, 7], torch.full((2,), -.6))
        hold.reset(torch.tensor([0]))
        torch.testing.assert_close(hold.effort[:, 0], torch.tensor([0., -.6]))

    def make_reward_inputs(self):
        angle = math.radians(6)
        state = types.SimpleNamespace(
            gravity=torch.tensor([[0., 0., -1.], [0., -math.sin(angle), -math.cos(angle)],
                                  [math.sin(angle), 0., -math.cos(angle)]]),
            height=torch.ones(3), linear_velocity=torch.zeros(3, 3), angular_velocity=torch.zeros(3, 3),
            vertical_velocity=torch.zeros(3), wheel_clearance=torch.zeros(3, 2),
            wheel_contact=torch.ones(3, 2, dtype=torch.bool), finite=torch.ones(3, dtype=torch.bool))
        commands = types.SimpleNamespace(velocity=torch.zeros(3, 3), height=torch.ones(3))
        motors = types.SimpleNamespace(effort=torch.zeros(3, 6), actions=torch.zeros(3, 6), previous=torch.zeros(3, 6))
        jump = types.SimpleNamespace(IDLE=0, PUSH=1, FLIGHT=2, LAND=3, phase=torch.zeros(3, dtype=torch.long),
            push_reference=lambda: (torch.zeros(3), torch.zeros(3)),
            air_now=torch.zeros(3, dtype=torch.bool), seen_air=torch.zeros(3, dtype=torch.bool),
            exit_event=torch.zeros(3, dtype=torch.bool), success_event=torch.zeros(3, dtype=torch.bool),
            peak=torch.zeros(3), target_peak=torch.zeros(3))
        cfg = types.SimpleNamespace(weights=dict(roll_error=-20., pitch_error=-10., roll_pitch_rate=-.2))
        return RewardModel(cfg, 3, "cpu"), state, commands, motors, jump

    def test_posture_reward_sign_phase_and_time_scaling(self):
        reward, state, commands, motors, jump = self.make_reward_inputs()
        angle2 = math.radians(6)**2
        result = reward.compute(state, commands, motors, jump, torch.zeros(3, dtype=torch.bool), .02)
        torch.testing.assert_close(result, torch.tensor([0., -20*angle2*.02, -10*angle2*.02]))
        jump.phase[:] = jump.FLIGHT
        result = reward.compute(state, commands, motors, jump, torch.zeros(3, dtype=torch.bool), .02)
        torch.testing.assert_close(result, torch.zeros(3))
        jump.phase[:] = jump.LAND
        state.angular_velocity[:, :2] = 1
        result = reward.compute(state, commands, motors, jump, torch.zeros(3, dtype=torch.bool), .01)
        torch.testing.assert_close(result, torch.tensor([-.4, -20*angle2-.4, -10*angle2-.4]) * .01)

    def test_telemetry_uses_measured_time_and_subset_reset(self):
        reward, state, commands, motors, jump = self.make_reward_inputs()
        reward.compute(state, commands, motors, jump, torch.zeros(3, dtype=torch.bool), .02)
        logs = reward.reset_logs(torch.tensor([1]), torch.tensor([10.]))
        self.assertAlmostEqual(float(logs['Episode/Posture/RollRMSDeg']), 6., places=4)
        self.assertEqual(float(logs['Episode/Posture/BothWheelsContactFraction']), 1.)
        self.assertEqual(float(reward.metric_time[1]), 0.)
        self.assertGreater(float(reward.metric_time[2]), 0.)
        state.finite[:] = False
        state.gravity[:] = float('nan')
        result = reward.compute(state, commands, motors, jump, torch.zeros(3, dtype=torch.bool), .02)
        torch.testing.assert_close(result, torch.zeros(3))
        logs = reward.reset_logs(torch.tensor([2]), torch.tensor([10.]))
        self.assertAlmostEqual(float(logs['Episode/Posture/PitchRMSDeg']), 6., places=4)

    def test_stance_alignment_and_support_apply_only_to_grounded_standing(self):
        _, state, commands, motors, jump = self.make_reward_inputs()
        reward = RewardModel(types.SimpleNamespace(weights=dict(wheel_stagger=-50., double_support=.5)), 3, 'cpu')
        state.wheel_fore_aft_difference = torch.tensor([.2, .1, .3])
        state.wheel_contact[1, 1] = False
        jump.phase[2] = jump.FLIGHT
        result = reward.compute(state, commands, motors, jump, torch.zeros(3, dtype=torch.bool), .02)
        torch.testing.assert_close(result, torch.tensor([-.03, -.01, 0.]))
        commands.velocity[:, 0] = 1.0
        result = reward.compute(state, commands, motors, jump, torch.zeros(3, dtype=torch.bool), .02)
        torch.testing.assert_close(result, torch.zeros(3))

    def test_clearance_and_non_wheel_contact_do_not_reward_single_wheel_balance(self):
        _, state, commands, motors, jump = self.make_reward_inputs()
        reward = RewardModel(types.SimpleNamespace(weights=dict(standing_wheel_clearance=-80., non_wheel_contact=-2.)), 3, 'cpu')
        state.wheel_clearance = torch.tensor([[.1, 0.], [0., .1], [.1, .1]])
        state.non_wheel_contact = torch.tensor([False, True, True])
        jump.phase[2] = jump.FLIGHT
        result = reward.compute(state, commands, motors, jump, torch.zeros(3, dtype=torch.bool), .02)
        torch.testing.assert_close(result, torch.tensor([-.8, -2.8, -2.]) * .02)

    def test_control_change_exception_keeps_action_and_history_checks(self):
        spec = RobotDescription.load(ROOT / 'src/wheel_legged_lab/assets/v2_0.json')
        old = observation_schema(spec)
        old['observation_settings'] = dict(actor_history=1, actor_delay=0)
        new = deepcopy(old)
        new['observation_settings']['actor_delay'] = 1
        new['motor_settings'] = dict(velocity_feedback='finite_difference')
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model.pt'
            save_schema(directory, old)
            with self.assertRaises(ValueError):
                validate_checkpoint(checkpoint, new)
            with self.assertWarns(UserWarning):
                validate_checkpoint(checkpoint, new, allow_control_change=True)
            new['actor_dim'] = 105
            with self.assertRaises(ValueError):
                validate_checkpoint(checkpoint, new, allow_control_change=True)

    def test_auxiliary_change_exception_still_rejects_layout_changes(self):
        spec = RobotDescription.load(ROOT / "src/wheel_legged_lab/assets/v2_0.json")
        old = observation_schema(spec)
        new = deepcopy(old)
        new['auxiliary_holds'] = {name: asdict(hold) for name, hold in spec.auxiliary_holds.items()}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model.pt'
            save_schema(directory, old)
            with self.assertRaisesRegex(ValueError, 'auxiliary control'):
                validate_checkpoint(checkpoint, new)
            with self.assertWarns(UserWarning):
                validate_checkpoint(checkpoint, new, allow_auxiliary_change=True)
            invalid = deepcopy(new)
            invalid['action_directions'][-1] = 1
            with self.assertRaisesRegex(ValueError, 'schema does not match'):
                validate_checkpoint(checkpoint, invalid, allow_auxiliary_change=True)
            save_schema(directory, new)
            validate_checkpoint(checkpoint, new)


if __name__ == '__main__':
    unittest.main()
