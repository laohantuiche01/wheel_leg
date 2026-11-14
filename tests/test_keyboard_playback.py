"""Keyboard edge semantics, task isolation, focus lifecycle and command ramp integration."""
from pathlib import Path
import sys
import types
import json
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType('wheel_legged_lab')
package.__path__ = [str(ROOT/'src/wheel_legged_lab')]
sys.modules.setdefault('wheel_legged_lab', package)
from wheel_legged_lab.settings import settings_for
from wheel_legged_lab.components.keyboard import KeyboardCommandState, PlaybackKeyboard
from wheel_legged_lab.components.locomotion import LocomotionCommands


class KeyboardPlaybackTest(unittest.TestCase):
    def state(self, task='walk', **kwargs):
        return KeyboardCommandState(settings_for(task), .28 if task in ('height','jump') else .35, **kwargs)

    def test_opposite_keys_and_repeat_do_not_latch_or_accumulate(self):
        state = self.state()
        state.handle_key('W', True)
        state.handle_key('W', True)
        state.handle_key('W', True, repeat=True)
        self.assertEqual(state.advance(.02).vx, 2.)
        state.handle_key('S', True)
        self.assertEqual(state.advance(.02).vx, 0.)
        state.handle_key('S', False)
        self.assertEqual(state.advance(.02).vx, 2.)
        state.handle_key('W', False)
        self.assertEqual(state.advance(.02).vx, 0.)
        state.handle_key('W', True, repeat=True)
        self.assertEqual(state.advance(.02).vx, 0.)

    def test_task_axes_and_saved_training_limits(self):
        walk = self.state(speed=10.)
        walk.handle_key('W', True)
        walk.handle_key('A', True)
        cmd = walk.advance(.02)
        self.assertEqual((cmd.vx,cmd.wz),(2.,0.))
        spin = self.state('spin', turn_rate=10.)
        spin.handle_key('S', True)
        spin.handle_key('D', True)
        cmd = spin.advance(.02)
        self.assertEqual((cmd.vx,cmd.wz),(0.,-2.))
        saved = settings_for('walk')
        saved.commands.vx, saved.commands.wz = (-1.,1.),(-1.2,1.2)
        old = KeyboardCommandState(saved,.35)
        old.handle_key('W',True)
        old.handle_key('A',True)
        cmd = old.advance(.02)
        self.assertEqual((cmd.vx,cmd.wz),(1.,1.2))

    def test_press_release_in_one_tick_survives_and_expires(self):
        state = self.state(pulse_hold=.2)
        for _ in range(4):
            state.handle_key('W',True)
            state.handle_key('W',False)
        for _ in range(10):
            self.assertEqual(state.advance(.02).vx,2.)
        self.assertEqual(state.advance(.02).vx,0.)
        # A real held key observed by the policy still stops on release.
        state.handle_key('W',True)
        self.assertEqual(state.advance(.02).vx,2.)
        state.handle_key('W',False)
        self.assertEqual(state.advance(.02).vx,0.)

    def test_character_stream_refreshes_pulses_and_reset_clears(self):
        state = self.state('spin',pulse_hold=.2)
        state.handle_character('a')
        for _ in range(8):
            self.assertEqual(state.advance(.02).wz,2.)
        state.handle_character('A')
        for _ in range(10):
            self.assertEqual(state.advance(.02).wz,2.)
        self.assertEqual(state.advance(.02).wz,0.)
        state.handle_character('a')
        state.release_all()
        self.assertEqual(state.advance(.02).wz,0.)
        state.handle_character('a')
        state.handle_key('L',True)
        self.assertEqual(state.advance(.02).wz,0.)
        jump = self.state('jump')
        jump.handle_character('q')
        self.assertFalse(jump.advance(.02).jump)
        strict = self.state(pulse_hold=0.)
        strict.handle_key('W',True)
        strict.handle_key('W',False)
        strict.handle_character('w')
        self.assertEqual(strict.advance(.02).vx,0.)

    def test_height_integration_hold_reset_and_fixed_height(self):
        state = self.state('height')
        state.handle_key('Z',True)
        self.assertAlmostEqual(state.advance(1.).height,.34)
        state.handle_key('Z',False)
        self.assertAlmostEqual(state.advance(1.).height,.34)
        state.handle_key('X',True)
        self.assertAlmostEqual(state.advance(2.).height,.20)
        state.release_all()
        self.assertAlmostEqual(state.advance(.1).height,.20)
        state.handle_key('L',True)
        self.assertAlmostEqual(state.advance(.1).height,.28)
        walk = self.state()
        walk.handle_key('Z',True)
        self.assertEqual(walk.advance(5.).height,.35)
        fixed = self.state('height',fixed_height=.31)
        fixed.handle_key('X',True)
        self.assertEqual(fixed.advance(5.).height,.31)

    def test_jump_is_edge_triggered_and_only_enabled_for_jump(self):
        state = self.state('jump')
        state.handle_key('Q',True)
        self.assertTrue(state.advance(.02).jump)
        state.handle_key('Q',True)
        state.handle_key('Q',True,repeat=True)
        self.assertFalse(state.advance(.02).jump)
        state.handle_key('Q',False)
        state.handle_key('Q',True)
        self.assertTrue(state.advance(.02).jump)
        state.handle_key('Q',False)
        state.handle_key('Q',True)
        state.reset()
        self.assertFalse(state.advance(.02).jump)
        other = self.state('spin')
        other.handle_key('Q',True)
        self.assertFalse(other.advance(.02).jump)

    def test_cli_priority_and_invalid_configuration(self):
        state = self.state(fixed_vx=.75,fixed_wz=0.)
        state.handle_key('S',True)
        self.assertEqual(state.advance(.02).vx,.75)
        state.reset()
        self.assertEqual(state.advance(.02).vx,.75)
        for kwargs in ({'speed':0.},{'turn_rate':float('nan')},{'fixed_vx':float('inf')},
                       {'pulse_hold':-.1},{'pulse_hold':float('nan')}):
            with self.assertRaises(ValueError):
                self.state(**kwargs)
        with self.assertRaises(ValueError):
            self.state('spin',fixed_vx=1.)
        with self.assertRaises(ValueError):
            self.state(fixed_wz=1.)
        with self.assertRaises(ValueError):
            state.advance(-.02)

    def test_manual_commands_use_existing_ramp_and_release_targets_zero(self):
        settings = settings_for('walk')
        state = KeyboardCommandState(settings,.35)
        sampler = LocomotionCommands(settings.commands,1,'cpu')
        ids = torch.arange(1)
        sampler.reset(ids)
        state.handle_key('W',True)
        for _ in range(120):
            cmd = state.advance(.02)
            sampler.set_manual(ids,vx=cmd.vx,wz=cmd.wz)
            sampler.advance(.02)
        before = float(sampler.velocity[0,0])
        state.handle_key('W',False)
        cmd = state.advance(.02)
        sampler.set_manual(ids,vx=cmd.vx,wz=cmd.wz)
        self.assertEqual(float(sampler.sampler.manual_velocity[0,0]),0.)
        sampler.advance(.02)
        self.assertAlmostEqual(float(sampler.velocity[0,0]),before-.016,places=5)
        sampler.reset(ids)
        for _ in range(100):
            sampler.advance(.02)
        self.assertEqual(float(sampler.velocity[0,0]),0.)

    def adapter(self, task='walk', log_path=None):
        event_types = types.SimpleNamespace(KEY_PRESS=1,KEY_RELEASE=2,KEY_REPEAT=3,CHAR=4)
        callbacks, unsubscribed, writes = [],[],[]
        interface = types.SimpleNamespace(subscribe_to_keyboard_events=lambda keyboard,callback:callbacks.append(callback) or 0,
            unsubscribe_to_keyboard_events=lambda keyboard,token:unsubscribed.append((keyboard,token)))
        window = types.SimpleNamespace(focused=True,get_keyboard=lambda:'keyboard')
        window.is_focused = lambda:window.focused
        carb, omni = types.ModuleType('carb'),types.ModuleType('omni')
        carb.__path__,omni.__path__ = [],[]
        carb.input = types.ModuleType('carb.input')
        carb.input.KeyboardEventType = event_types
        carb.input.acquire_input_interface = lambda:interface
        omni.appwindow = types.ModuleType('omni.appwindow')
        omni.appwindow.get_default_app_window = lambda:window
        env = types.SimpleNamespace(set_play_velocity_command=lambda **kwargs:writes.append(('velocity',kwargs)),
            set_height_command=lambda h:writes.append(('height',h)),request_jump_takeoff=lambda:writes.append(('jump',True)))
        with patch.dict(sys.modules,{'carb':carb,'carb.input':carb.input,'omni':omni,'omni.appwindow':omni.appwindow}):
            adapter = PlaybackKeyboard(env,self.state(task),log_path=log_path)
        return adapter,callbacks[0],window,writes,unsubscribed

    def test_omni_events_queue_focus_loss_clears_and_close_unsubscribes(self):
        adapter,callback,window,writes,unsubscribed = self.adapter()
        def event(key,kind):
            return types.SimpleNamespace(input=types.SimpleNamespace(name=key),type=kind)
        callback(event('W',1))
        self.assertEqual(writes,[])
        self.assertEqual(adapter.advance(.02).vx,2.)
        window.focused = False
        self.assertEqual(adapter.advance(.02).vx,0.)
        callback(event('W',1))
        window.focused = True
        callback(event('W',3))
        self.assertEqual(adapter.advance(.02).vx,0.)
        callback(event('W',1))
        self.assertEqual(adapter.advance(.02).vx,2.)
        callback(event('W',2))
        self.assertEqual(adapter.advance(.02).vx,0.)
        adapter.close()
        adapter.close()
        self.assertEqual(unsubscribed,[('keyboard',0)])

    def test_adapter_jump_is_applied_once_and_reset_clears_it(self):
        adapter,callback,_,writes,_ = self.adapter('jump')
        callback(types.SimpleNamespace(input='Q',type=1))
        adapter.advance(.02)
        adapter.advance(.02)
        self.assertEqual([w for w in writes if w[0]=='jump'],[('jump',True)])
        callback(types.SimpleNamespace(input='Q',type=2))
        callback(types.SimpleNamespace(input='Q',type=1))
        adapter.reset()
        adapter.advance(0.)
        self.assertEqual([w for w in writes if w[0]=='jump'],[('jump',True)])
        adapter.close()

    def test_native_short_pulses_and_character_input_are_traced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'keyboard.jsonl'
            adapter,callback,window,_,_ = self.adapter(log_path=path)
            callback(types.SimpleNamespace(input='W',type=1))
            callback(types.SimpleNamespace(input='W',type=2))
            callback(types.SimpleNamespace(input='w',type=4))
            self.assertEqual(adapter.advance(.02).vx,2.)
            window.focused = False
            self.assertEqual(adapter.advance(.02).vx,0.)
            callback(types.SimpleNamespace(input='w',type=4))
            window.focused = True
            self.assertEqual(adapter.advance(.02).vx,0.)
            adapter.close()
            adapter.close()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows[0]['kind'],'start')
            self.assertEqual(sum(row['kind']=='input' for row in rows),4)
            self.assertEqual(rows[-1]['kind'],'close')
            self.assertTrue(any(row.get('requested',[0])[0]==2. for row in rows))


if __name__ == '__main__':
    unittest.main()
