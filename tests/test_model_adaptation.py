"""CPU contract/control tests; bypass package registration and never import Isaac."""
from copy import deepcopy
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

from wheel_legged_lab.robot import RobotDescription, RobotBinding
from wheel_legged_lab.settings import MotorSettings, ObservationSettings, SpringSettings, settings_for
from wheel_legged_lab.components.motors import MotorBank, SpringController
from wheel_legged_lab.components.observations import ObservationBuilder, observation_schema
from wheel_legged_lab.checkpoints import save_schema, validate_checkpoint


class ModelAdaptationTest(unittest.TestCase):
    def setUp(self):
        self.spec = RobotDescription.load(ROOT / "src/wheel_legged_lab/assets/v2_0.json")

    def test_full_joint_ownership_and_reset(self):
        self.assertEqual(len(self.spec.joint_names), 30)
        self.spec.validate_joints(list(reversed(self.spec.joint_names)))
        self.assertEqual(set(self.spec.rest_joint_positions.values()), {0.0})
        for mutate in (
            lambda s: s.spring_joints.append("left_leg_1_joint"),
            lambda s: s.rest_joint_positions.pop("gimbal_pitch_joint"),
            lambda s: s.passive_joint_patterns.append(".*"),
        ):
            spec = deepcopy(self.spec)
            mutate(spec)
            with self.assertRaises(ValueError):
                spec.validate_joints(self.spec.joint_names)
        with self.assertRaises(ValueError):
            self.spec.validate_joints(self.spec.joint_names + ["unknown_joint"])

    def test_binding_uses_semantics_not_physics_order(self):
        bodies = self.spec.wheel_bodies + self.spec.termination_bodies
        robot = types.SimpleNamespace(joint_names=list(reversed(self.spec.joint_names)), body_names=bodies)
        sensor = types.SimpleNamespace(body_names=list(reversed(bodies)))
        binding = RobotBinding(robot, sensor, self.spec)
        self.assertEqual([robot.joint_names[i] for i in binding.action_ids], self.spec.action_names)
        self.assertEqual([sensor.body_names[i] for i in binding.wheel_sensor_ids], self.spec.wheel_bodies)
        self.assertNotEqual(binding.wheel_sensor_ids, binding.wheel_body_ids)

    def make_bank(self, spec=None):
        spec = spec or self.spec
        data = types.SimpleNamespace(joint_pos=torch.zeros(2, 6), joint_vel=torch.zeros(2, 6),
            default_joint_pos=torch.zeros(2, 6),
            soft_joint_pos_limits=torch.tensor([-2.0, 2.0]).repeat(2, 6, 1))
        written = torch.zeros(2, 6)
        def write(effort, joint_ids):
            written[:, joint_ids] = effort
        robot = types.SimpleNamespace(data=data, set_joint_effort_target=write)
        bank = MotorBank(robot, types.SimpleNamespace(action_ids=list(range(6))), spec, MotorSettings(), 2, "cpu")
        return bank, robot, written

    def test_direction_applies_to_torque_and_critic_effort(self):
        bank, robot, written = self.make_bank()
        bank.accept(torch.ones(2, 6))
        bank.apply()
        torch.testing.assert_close(written[0], torch.tensor([30., 30., -30., -30., 2., -2.]))
        torch.testing.assert_close(bank.effort[0], torch.tensor([30., 30., 30., 30., 2., 2.]))
        # A unit policy offset maps to +/-0.5 rad and +/-10 rad/s physically.
        robot.data.joint_pos[:, :4] = torch.tensor([.5, .5, -.5, -.5])
        robot.data.joint_vel[:, 4:] = torch.tensor([10., -10.])
        bank.apply()
        torch.testing.assert_close(written, torch.zeros_like(written))

    def test_observation_coordinates_round_trip(self):
        bank, robot, _ = self.make_bank()
        robot.data.joint_pos[:] = torch.tensor([.1, .1, -.1, -.1, 99., -99.])
        robot.data.joint_vel[:] = bank.directions * 2
        commands = types.SimpleNamespace(velocity=torch.zeros(2, 3), height=torch.ones(2))
        jump = types.SimpleNamespace(observation=lambda: torch.zeros(2, 7))
        builder = ObservationBuilder(ObservationSettings(noise_enabled=False), robot, bank, commands, jump, None, None)
        state = types.SimpleNamespace(angular_velocity=torch.zeros(2, 3), gravity=torch.zeros(2, 3))
        obs = builder._actor(state, noisy=False)
        self.assertEqual(tuple(obs.shape), (2, 35))
        torch.testing.assert_close(obs[0, 10:16], torch.tensor([.1, .1, .1, .1, 0., 0.]))
        torch.testing.assert_close(obs[0, 16:22], torch.full((6,), .1))

    def test_model_specific_standing_height_and_legacy_fallback(self):
        for task in ("stand", "height", "velocity", "jump"):
            settings_for(task).validate()
        self.assertEqual(settings_for("stand").commands.height, (self.spec.rest_height,) * 2)
        old = ROOT / "src/wheel_legged_lab/assets/v1_0.json"
        self.assertEqual(settings_for("stand", str(old)).commands.height, (.309195, .309195))
        bank, _, written = self.make_bank(RobotDescription.load(old))
        bank.accept(torch.ones(2, 6))
        bank.apply()
        torch.testing.assert_close(written[0], torch.tensor([30., 30., 30., 30., 2., 2.]))

    def test_spring_preload_tracks_each_models_zero(self):
        for rest in (0.0, -.0482):
            _, robot, written = self.make_bank()
            robot.data.default_joint_pos[:, :2] = rest
            robot.data.joint_pos[:, :2] = rest
            springs = SpringController(robot, types.SimpleNamespace(spring_ids=[0, 1]), SpringSettings(), 2, "cpu")
            springs.apply()
            torch.testing.assert_close(written[:, :2], torch.full((2, 2), 400.))
            robot.data.joint_pos[:, :2] -= .01
            springs.apply()
            torch.testing.assert_close(written[:, :2], torch.full((2, 2), 400. + 200. / 7.))

    def test_checkpoint_rejects_equal_dimensions_with_wrong_directions(self):
        schema = observation_schema(self.spec)
        self.assertEqual((schema["actor_dim"], schema["critic_dim"]), (35, 60))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            save_schema(directory, schema)
            validate_checkpoint(checkpoint, schema)
            changed = deepcopy(schema)
            changed["action_directions"][-1] = 1
            with self.assertRaises(ValueError):
                validate_checkpoint(checkpoint, changed)


if __name__ == "__main__":
    unittest.main()
