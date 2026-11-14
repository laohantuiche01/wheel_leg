# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
"""Measured in-place jump, adapted from the old jump_takeoff release/landing logic.

The legacy JumpController remains intact for historical checkpoint playback.
This component only schedules observations and rewards; it never writes forces,
velocities or joint targets.
"""
import torch
from dataclasses import asdict
from .jump import JumpController


def jump_task_schema(settings):
    values = asdict(settings.jump)
    values.pop('automatic')  # training vs explicit playback trigger, not observation semantics
    return dict(version='measured-jump-v1', commands=asdict(settings.commands), jump=values)


class MeasuredJumpController(JumpController):
    measured = True

    def __init__(self, cfg, n, device):
        super().__init__(cfg, n, device)
        for name in ('release_event', 'failure_event', 'push_timeout_event'):
            setattr(self, name, torch.zeros(n, dtype=torch.bool, device=device))
        for name in ('age', 'push_progress', 'push_progress_reward', 'releases', 'failures',
                     'push_timeouts', 'no_air_timeouts', 'landing_timeouts', 'release_speed_sum',
                     'release_time_sum', 'release_clearance_sum', 'release_force_sum',
                     'landing_samples', 'stable_time_max'):
            setattr(self, name, torch.zeros(n, device=device))
        self.landing_blocked = torch.zeros(n, 5, device=device)
        self.age = self.age.double()
        self.ready_height_tolerance = cfg.height_tolerance

    def request(self, ids):
        if not self.cfg.enabled:
            raise RuntimeError('Jump is disabled for this task')
        # Busy requests are discarded: holding Q cannot queue a second jump.
        self.requested[ids] |= self.phase[ids] == self.IDLE

    def _finish(self, mask):
        self.exit_event |= mask
        self.phase[mask] = self.IDLE
        self.cooldown[mask] = self.cfg.cooldown_s
        self.requested[mask] = False

    def abort(self, terminated):
        """Count falls/resets as failed attempts, including same-tick landings."""
        mask = terminated & ((self.phase != self.IDLE) | self.success_event)
        self.successes -= (mask & self.success_event).float()
        self.success_event[mask] = False
        self.failure_event |= mask
        self.failures += mask.float()
        self._finish(mask)

    def advance(self, dt, state, commands, episode_time):
        c = self.cfg
        for name in ('trigger_event', 'takeoff_event', 'success_event', 'exit_event',
                     'release_event', 'failure_event', 'push_timeout_event', 'air_now'):
            getattr(self, name).zero_()
        self.push_progress_reward.zero_()
        if not c.enabled:
            return
        if dt <= 0:
            raise ValueError('Jump control dt must be positive')
        if state.wheel_contact_force is None:
            raise ValueError('Measured jump requires wheel contact forces')
        self.age += dt  # independent of randomized PPO initial episode ages
        self.cooldown.sub_(dt).clamp_(min=0)
        active = self.phase != self.IDLE
        pushing = self.phase == self.PUSH
        self.requested[active] = False
        self.elapsed[active] += dt
        self.peak[active] = torch.maximum(self.peak[active], state.height[active])
        self.max_upward_speed[pushing] = torch.maximum(self.max_upward_speed[pushing], state.vertical_velocity[pushing])
        progress = ((state.vertical_velocity - c.min_upward_speed) /
                    (self.release_velocity - c.min_upward_speed).clamp(min=1e-6)).clamp(0, 1)
        improved = torch.maximum(self.push_progress, progress)
        self.push_progress_reward[pushing] = (improved[pushing] - self.push_progress[pushing]) / dt
        self.push_progress[pushing] = improved[pushing]

        air = active & state.finite & (state.wheel_clearance > c.air_clearance).all(dim=1) & ~state.wheel_contact.any(dim=1)
        self.air_now.copy_(air)
        self.air_time.copy_(torch.where(air, self.air_time + dt, 0.))
        confirmed = air & (self.air_time + 1e-6 >= c.min_air_s) & (self.max_upward_speed >= c.min_upward_speed)
        self.takeoff_event.copy_(confirmed & ~self.seen_air)
        self.takeoffs += self.takeoff_event.float()
        self.seen_air |= confirmed
        force = state.wheel_contact_force.sum(dim=1)
        unloaded = force <= c.release_force_max
        required_speed = (self.release_velocity * c.release_speed_fraction).clamp(min=c.min_upward_speed)
        departing = (state.vertical_velocity >= required_speed) & ((state.height >= c.release_height) | unloaded)
        already_air = air & unloaded & (self.max_upward_speed >= c.min_upward_speed)
        self.release_event.copy_(pushing & (departing | already_air))
        self.phase[self.release_event] = self.FLIGHT
        self.releases += self.release_event.float()
        self.release_speed_sum += torch.where(self.release_event, state.vertical_velocity, 0.)
        self.release_time_sum += torch.where(self.release_event, self.elapsed, 0.)
        self.release_clearance_sum += torch.where(self.release_event, state.wheel_clearance.amin(dim=1), 0.)
        self.release_force_sum += torch.where(self.release_event, force, 0.)

        both = state.wheel_contact.all(dim=1)
        touching = active & self.seen_air & both
        self.phase[(self.phase == self.LAND) & ~both] = self.FLIGHT
        self.phase[touching & (state.vertical_velocity <= c.vertical_speed_tolerance)] = self.LAND
        gates = torch.stack(((state.height - commands.height).abs() <= c.height_tolerance,
            state.vertical_velocity.abs() <= c.vertical_speed_tolerance,
            -state.gravity[:, 2] >= c.tilt_cos_min,
            state.angular_velocity.norm(dim=1) <= c.angular_speed_tolerance,
            (state.linear_velocity[:, :2] - commands.velocity[:, :2]).norm(dim=1) <= c.horizontal_speed_tolerance), dim=1)
        self.landing_samples += touching.float()
        self.landing_blocked += (touching[:, None] & ~gates).float()
        valid_contact = ~state.forbidden_contact
        if state.non_wheel_contact is not None:
            valid_contact &= ~state.non_wheel_contact
        stable = touching & state.finite & valid_contact & gates.all(dim=1)
        self.stable_time.copy_(torch.where(stable, self.stable_time + dt, 0.))
        self.stable_time_max.copy_(torch.maximum(self.stable_time_max, self.stable_time))
        self.success_event.copy_(stable & (self.stable_time + 1e-6 >= c.stable_landing_s))
        self.successes += self.success_event.float()
        self.push_timeout_event.copy_((self.phase == self.PUSH) & (self.elapsed + 1e-6 >= c.push_timeout_s))
        failed = active & ~self.success_event & (self.push_timeout_event | (self.elapsed + 1e-6 >= c.timeout_s))
        self.failure_event.copy_(failed)
        self.failures += failed.float()
        self.push_timeouts += self.push_timeout_event.float()
        self.no_air_timeouts += (failed & ~self.seen_air).float()
        self.landing_timeouts += (failed & self.seen_air).float()
        self._finish(self.success_event | failed)

        ready_gates = gates.clone()
        ready_gates[:, 0] = (state.height - commands.height).abs() <= self.ready_height_tolerance
        ready = ((self.phase == self.IDLE) & (self.cooldown <= 0) & (self.age + 1e-6 >= c.min_episode_s)
            & both & state.finite & valid_contact & ready_gates.all(dim=1))
        if c.automatic:
            self.requested |= ready & (torch.rand_like(self.age) < min(c.rate_per_s * dt, 1.))
        enter = ready & self.requested & ~self.exit_event
        self.trigger_event.copy_(enter)
        self.attempts += enter.float()
        self.phase[enter] = self.PUSH
        self.requested[enter] = False
        for name in ('elapsed', 'air_time', 'stable_time', 'push_progress', 'max_upward_speed'):
            getattr(self, name)[enter] = 0
        self.seen_air[enter] = False
        self.start_height[enter] = state.height[enter]
        self.peak[enter] = state.height[enter]
        self.target_peak[enter] = torch.empty_like(self.target_peak[enter]).uniform_(*c.peak_height)
        self.release_velocity[enter] = (2 * 9.81 * (self.target_peak[enter] - c.release_height)).clamp(min=0).sqrt()
        duration = 2 * (c.release_height - self.start_height[enter]).clamp(min=.01) / self.release_velocity[enter].clamp(min=.1)
        self.push_duration[enter] = duration.clamp(min=dt, max=c.max_push_s)

    def observation(self):
        # LAND uses the pretrained standing controller, while internal landing
        # checks continue. Seven slots and the base network width are retained.
        control_jump = (self.phase == self.PUSH) | (self.phase == self.FLIGHT)
        flags = torch.zeros(len(self.phase), 5, device=self.phase.device)
        flags[:, 0] = (~control_jump).float()
        flags[:, 4] = control_jump.float()
        return torch.cat((flags, torch.where(control_jump, self.target_peak, 0.)[:, None],
            torch.where(control_jump, self.elapsed / self.cfg.timeout_s, 0.)[:, None]), dim=1)

    def transition_snapshot(self, state):
        data = {name: getattr(self, name).clone() for name in ('phase', 'elapsed', 'trigger_event',
            'release_event', 'takeoff_event', 'success_event', 'failure_event', 'exit_event',
            'push_timeout_event', 'peak', 'target_peak', 'stable_time')}
        data.update(height=state.height.clone(), vertical_velocity=state.vertical_velocity.clone(),
            clearance=state.wheel_clearance.clone(), contact=state.wheel_contact.clone(),
            finite=state.finite.clone(), control_observation=self.observation().clone())
        return data

    def episode_logs(self, ids):
        attempts = self.attempts[ids].sum().clamp(min=1)
        release_count = self.releases[ids].sum().clamp(min=1)
        samples = self.landing_samples[ids].sum().clamp(min=1)
        logs = {f'Episode/Jump/{name}': getattr(self, field)[ids].sum() / attempts for name, field in (
            ('ReleasePerAttempt', 'releases'), ('FailurePerAttempt', 'failures'),
            ('PushTimeoutPerAttempt', 'push_timeouts'), ('NoAirTimeoutPerAttempt', 'no_air_timeouts'),
            ('LandingTimeoutPerAttempt', 'landing_timeouts'))}
        for name, field in (('ReleaseVelZMean', 'release_speed_sum'), ('ReleaseTimeMean', 'release_time_sum'),
                            ('ReleaseClearanceMean', 'release_clearance_sum'), ('ReleaseContactForceMean', 'release_force_sum')):
            logs[f'Episode/Jump/{name}'] = getattr(self, field)[ids].sum() / release_count
        for i, name in enumerate(('Height', 'VelZ', 'Tilt', 'AngVel', 'LinVel')):
            logs[f'Episode/Jump/Landing{name}BlockedFraction'] = self.landing_blocked[ids, i].sum() / samples
        logs['Episode/Jump/LandingContactSamplesMean'] = self.landing_samples[ids].mean()
        logs['Episode/Jump/StableTimeMaxMean'] = self.stable_time_max[ids].mean()
        return logs
