"""Unified playback commands, enabled explicitly after checkpoint validation.

Training configuration and the saved checkpoint contract are never mutated.
Only command/phase inputs change; the loaded policy and motor controllers remain
in charge of the physical actions.
"""
from copy import deepcopy
import math
from ..settings import JumpSettings
from .jump import JumpController
from .jump_task import MeasuredJumpController
from .locomotion import LocomotionCommands


class UnifiedKeyboardControl:
    def __init__(self, env, height_range=(.20, .38)):
        if (len(height_range) != 2 or not all(math.isfinite(x) for x in height_range)
                or not 0 < height_range[0] < height_range[1]):
            raise ValueError('Keyboard height range must be finite, positive and increasing')
        self.env, self.height_range = env, tuple(height_range)
        trained = env.cfg.settings
        if trained.jump.enabled:
            cfg = deepcopy(trained.jump)
            base_height = sum(trained.commands.height) / 2
        else:
            cfg = JumpSettings(enabled=True, implementation='measured_v2')
            base_height = .28
        cfg.enabled, cfg.automatic = True, False
        self.release_offset = max(cfg.release_height - base_height, .02)
        self.peak_offsets = (max(cfg.peak_height[0] - base_height, self.release_offset + .02),
                             max(cfg.peak_height[1] - base_height, self.release_offset + .04))
        controller = MeasuredJumpController if cfg.implementation.startswith('measured_') else JumpController
        env.jump = controller(cfg, env.num_envs, env.device)
        env.commands = LocomotionCommands(deepcopy(trained.commands), env.num_envs, env.device)
        # ObservationBuilder stores references to these components.
        env.observations.jump = env.jump
        env.observations.commands = env.commands

    def set_velocity(self, ids, vx=None, vy=None, wz=None):
        # Bypass only task-axis exclusions. CommandSampler still checks finite
        # values and rejects lateral motion; the ordinary slew limiter is kept.
        self.env.commands.sampler.set_manual(ids, vx=vx, vy=vy, wz=wz)

    def set_height(self, ids, height):
        if not math.isfinite(height) or not self.height_range[0] <= height <= self.height_range[1]:
            raise ValueError(f'Keyboard target height must lie in {self.height_range}')
        self.env.commands.sampler.set_manual(ids, height=height)
        jump = self.env.jump
        if bool((jump.phase == jump.IDLE).all()):
            # All keyboard environments receive the same height. Freeze the
            # references during an attempt, so Z/X cannot rewrite its apex.
            jump.cfg.release_height = height + self.release_offset
            jump.cfg.peak_height = tuple(height + offset for offset in self.peak_offsets)

    def request_jump(self, ids):
        self.env.jump.request(ids)
