"""All keys reach runtime commands for every task without changing training settings."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys,types,unittest
import torch
ROOT=Path(__file__).resolve().parents[1]
package=types.ModuleType('wheel_legged_lab');package.__path__=[str(ROOT/'src/wheel_legged_lab')]
sys.modules.setdefault('wheel_legged_lab',package)
from wheel_legged_lab.settings import settings_for,TASKS
from wheel_legged_lab.components.keyboard import KeyboardCommandState
from wheel_legged_lab.components.keyboard_mode import UnifiedKeyboardControl
from wheel_legged_lab.components.jump import JumpController


class UnifiedKeyboardTest(unittest.TestCase):
    def state(self,task,**kwargs):
        initial=.28 if task=='jump' else .35
        return KeyboardCommandState(settings_for(task),initial,unified=True,**kwargs)

    def test_every_task_enables_all_motion_height_and_jump_keys(self):
        for task in TASKS:
            with self.subTest(task=task):
                s=self.state(task)
                for key in ('W','A','Z','Q'):s.handle_key(key,True)
                c=s.advance(.02)
                self.assertEqual((c.vx,c.wz),(2.,2.));self.assertTrue(c.jump)
                self.assertAlmostEqual(c.height,s.default_height+.004)
                s.reset()
                for key in ('S','D','X'):s.handle_key(key,True)
                c=s.advance(.02)
                self.assertEqual((c.vx,c.wz),(-2.,-2.))
                self.assertAlmostEqual(c.height,s.default_height-.004)

    def test_cli_targets_are_initial_values_and_keys_take_over(self):
        s=self.state('jump',fixed_vx=.4,fixed_wz=-.3,fixed_height=.31)
        c=s.advance(0.);self.assertEqual((c.vx,c.wz,c.height),(.4,-.3,.31))
        s.handle_key('W',True);c=s.advance(.02)
        self.assertEqual((c.vx,c.wz),(2.,-.3))
        s.handle_key('W',False);self.assertEqual(s.advance(.02).vx,0.)
        s.handle_key('A',True);s.handle_key('Z',True);c=s.advance(.02)
        self.assertEqual(c.wz,2.);self.assertGreater(c.height,.31)
        s.handle_key('L',True);c=s.advance(.02)
        self.assertEqual((c.vx,c.wz,c.height),(0.,0.,.31))
        s=self.state('walk',fixed_vx=.4,fixed_wz=.3)
        s.release_all();self.assertEqual((s.advance(0.).vx,s.advance(0.).wz),(0.,0.))

    def test_unified_height_bounds_and_configurable_speeds(self):
        s=self.state('spin',speed=3.,turn_rate=4.,height_range=(.22,.4))
        for key in ('W','A','Z'):s.handle_key(key,True)
        c=s.advance(10.);self.assertEqual((c.vx,c.wz,c.height),(3.,4.,.4))
        s.handle_key('Z',False);s.handle_key('X',True)
        self.assertEqual(s.advance(10.).height,.22)
        for bounds in ((.4,.2),(0.,.3),(.2,float('nan'))):
            with self.assertRaises(ValueError):self.state('walk',height_range=bounds)
        with self.assertRaises(ValueError):self.state('walk',fixed_height=.5)

    def env(self,task):
        settings=settings_for(task)
        env=types.SimpleNamespace(cfg=types.SimpleNamespace(settings=settings),num_envs=2,device='cpu',
            jump=JumpController(settings.jump,2,'cpu'),observations=types.SimpleNamespace())
        return env,UnifiedKeyboardControl(env)

    def test_runtime_routes_mixed_commands_and_jump_for_every_task(self):
        for task in TASKS:
            with self.subTest(task=task):
                env,control=self.env(task);before=deepcopy(asdict(env.cfg.settings))
                ids=torch.arange(2);env.commands.reset(ids);env.jump.reset(ids)
                self.assertIs(env.observations.commands,env.commands);self.assertIs(env.observations.jump,env.jump)
                control.set_velocity(ids,vx=.5,vy=0.,wz=.4)
                control.set_height(ids,.3)
                for _ in range(120):env.commands.advance(.02)
                torch.testing.assert_close(env.commands.velocity,torch.tensor([[.5,0.,.4],[.5,0.,.4]]))
                env.jump.age[:]=3.
                state=types.SimpleNamespace(height=torch.full((2,),.3),vertical_velocity=torch.zeros(2),
                    wheel_clearance=torch.zeros(2,2),wheel_contact=torch.ones(2,2,dtype=torch.bool),
                    wheel_contact_force=torch.full((2,2),50.),finite=torch.ones(2,dtype=torch.bool),
                    forbidden_contact=torch.zeros(2,dtype=torch.bool),non_wheel_contact=torch.zeros(2,dtype=torch.bool),
                    gravity=torch.tensor([[0.,0.,-1.],[0.,0.,-1.]]),angular_velocity=torch.zeros(2,3),
                    linear_velocity=torch.tensor([[.5,0.,0.],[.5,0.,0.]]))
                control.request_jump(ids)
                env.jump.advance(.02,state,env.commands,torch.full((2,),3.))
                self.assertTrue((env.jump.phase==env.jump.PUSH).all())
                self.assertEqual(asdict(env.cfg.settings),before)

    def test_jump_references_follow_idle_height_and_freeze_during_attempt(self):
        env,control=self.env('jump');ids=torch.arange(2);env.commands.reset(ids)
        control.set_height(ids,.31)
        self.assertAlmostEqual(env.jump.cfg.release_height,.37)
        self.assertAlmostEqual(env.jump.cfg.peak_height[0],.43)
        env.jump.phase[:]=env.jump.PUSH
        control.set_height(ids,.33)
        self.assertAlmostEqual(env.jump.cfg.release_height,.37)
        env.jump.phase[:]=env.jump.IDLE
        control.set_height(ids,.33)
        self.assertAlmostEqual(env.jump.cfg.release_height,.39)
        with self.assertRaises(ValueError):control.set_height(ids,.5)
        with self.assertRaises(ValueError):control.set_velocity(ids,vy=.1)
        with self.assertRaises(ValueError):control.set_velocity(ids,vx=float('nan'))

if __name__=='__main__':unittest.main()
