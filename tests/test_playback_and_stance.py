"""Playback metadata resolution and body-frame stance geometry on CPU."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
package = types.ModuleType('wheel_legged_lab')
package.__path__ = [str(ROOT / 'src/wheel_legged_lab')]
sys.modules.setdefault('wheel_legged_lab', package)
from playback_config import resolve_playback_args, restore_runtime_settings
from wheel_legged_lab.settings import settings_for
from wheel_legged_lab.components.state import wheel_fore_aft_difference


class PlaybackAndStanceTest(unittest.TestCase):
    def args(self, path, **overrides):
        values = dict(checkpoint=path, task=None, strategy=None, actor_history=None, seed=None, robot_spec=None)
        values.update(overrides)
        return argparse.Namespace(**values)

    def fixture(self, directory):
        path = Path(directory) / 'model.pt'
        path.touch()
        (path.parent / 'launch.json').write_text(json.dumps(dict(task='stand', strategy='legacy_ppo', seed=42)))
        (path.parent / 'observation_schema.json').write_text(json.dumps(dict(observation_settings=dict(actor_history=3))))
        return path

    def test_saved_config_selected_without_simulator(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(self.fixture(directory))
            resolve_playback_args(args)
            self.assertEqual((args.task, args.strategy, args.actor_history, args.seed), ('stand', 'legacy_ppo', 3, 42))

    def test_explicit_mismatch_and_missing_checkpoint_fail_early(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, 'differs from training'):
                resolve_playback_args(self.args(path, strategy='baseline'))
            with self.assertRaisesRegex(ValueError, 'actor_history'):
                resolve_playback_args(self.args(path, actor_history=1))
            with self.assertRaises(FileNotFoundError):
                resolve_playback_args(self.args(Path(directory) / 'missing.pt'))

    def test_old_run_without_strategy_defaults_to_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(directory)
            (path.parent/'launch.json').write_text(json.dumps(dict(task='stand', seed=7)))
            (path.parent/'observation_schema.json').write_text('{}')
            args = self.args(path)
            resolve_playback_args(args)
            self.assertEqual((args.strategy, args.actor_history, args.seed), ('baseline', 1, 7))

    def test_runtime_controller_snapshot_wins_over_changed_defaults(self):
        saved = settings_for('stand', strategy='legacy_ppo')
        saved.motors.velocity_filter_alpha = .3
        saved.observations.actor_history = 3
        target = settings_for('stand', strategy='legacy_ppo')
        restore_runtime_settings(target, dict(motor_settings=asdict(saved.motors),
            observation_settings=asdict(saved.observations), spring_settings=asdict(saved.spring)))
        self.assertEqual(target.motors.velocity_filter_alpha, .3)
        self.assertEqual(target.observations.actor_history, 3)
        self.assertFalse(target.observations.noise_enabled)

    def test_legacy_mixed_walk_and_spin_ranges_restore_for_playback(self):
        saved = settings_for('walk')
        saved.commands.wz = (-2., 2.)
        saved.commands.straight_fraction = .6
        saved.rewards.lean_in_turns = True
        target = settings_for('walk')
        self.assertEqual(target.commands.wz, (0., 0.))
        restore_runtime_settings(target, dict(motion_task=dict(commands=asdict(saved.commands),
            spin=asdict(saved.spin), lean_in_turns=True)))
        self.assertEqual(target.commands.wz, (-2., 2.))
        self.assertTrue(target.rewards.lean_in_turns)
        spin = settings_for('spin')
        saved.commands = settings_for('spin').commands
        saved.commands.wz = (-3., 3.)
        restore_runtime_settings(spin, dict(motion_task=dict(commands=asdict(saved.commands))))
        self.assertEqual(spin.commands.wz, (-3., 3.))

    def test_fore_aft_offset_uses_body_frame_and_not_world_x(self):
        quats = torch.tensor([[1., 0., 0., 0.], [math.sqrt(.5), 0., 0., math.sqrt(.5)]])
        wheels = torch.tensor([[[.1, .2, 0.], [0., -.2, 0.]], [[-.4, .1, 0.], [0., 0., 0.]]])
        torch.testing.assert_close(wheel_fore_aft_difference(quats, wheels), torch.tensor([.1, .1]), atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
