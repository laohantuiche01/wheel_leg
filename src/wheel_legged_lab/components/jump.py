"""Independent jump lifecycle. No environment attributes or legacy state-machine hooks."""
import torch

class JumpController:
    IDLE, PUSH, FLIGHT, LAND = 0, 1, 2, 3

    def __init__(self, cfg, n, device):
        self.cfg = cfg
        self.phase = torch.zeros(n, dtype=torch.long, device=device)
        self.requested = torch.zeros(n, dtype=torch.bool, device=device)
        self.seen_air = torch.zeros_like(self.requested)
        self.air_now = torch.zeros_like(self.requested)
        self.trigger_event = torch.zeros_like(self.requested)
        self.takeoff_event = torch.zeros_like(self.requested)
        self.success_event = torch.zeros_like(self.requested)
        self.exit_event = torch.zeros_like(self.requested)
        for name in ("elapsed", "cooldown", "start_height", "target_peak", "peak", "max_upward_speed",
                     "air_time", "stable_time", "attempts", "takeoffs", "successes", "release_velocity", "push_duration"):
            setattr(self, name, torch.zeros(n, device=device))

    def reset(self, ids):
        for value in vars(self).values():
            if isinstance(value, torch.Tensor):
                value[ids] = 0

    def request(self, ids):
        if not self.cfg.enabled:
            raise RuntimeError("Jump is disabled for this task")
        self.requested[ids] = True

    def advance(self, dt, state, commands, episode_time):
        c = self.cfg
        for event in (self.trigger_event, self.takeoff_event, self.success_event, self.exit_event, self.air_now):
            event.zero_()
        if not c.enabled:
            return
        self.cooldown.sub_(dt).clamp_(min=0)
        active = self.phase != self.IDLE
        self.elapsed[active] += dt
        self.peak[active] = torch.maximum(self.peak[active], state.height[active])
        pushing = self.phase == self.PUSH
        self.max_upward_speed[pushing] = torch.maximum(self.max_upward_speed[pushing], state.vertical_velocity[pushing])
        air = active & (state.wheel_clearance > c.air_clearance).all(dim=1) & ~state.wheel_contact.any(dim=1)
        self.air_now.copy_(air)
        self.air_time.copy_(torch.where(air, self.air_time + dt, 0.0))
        confirmed = air & (self.air_time >= c.min_air_s) & (self.max_upward_speed >= c.min_upward_speed)
        self.takeoff_event.copy_(confirmed & ~self.seen_air)
        self.takeoffs += self.takeoff_event.float()
        self.seen_air |= confirmed
        release = pushing & (self.seen_air | (self.elapsed >= self.push_duration))
        self.phase[release] = self.FLIGHT
        touching = active & self.seen_air & state.wheel_contact.all(dim=1)
        self.phase[touching] = self.LAND
        stable = (touching & state.finite & ~state.forbidden_contact
                  & (state.vertical_velocity.abs() <= c.vertical_speed_tolerance)
                  & ((state.height - commands.height).abs() <= c.height_tolerance)
                  & (-state.gravity[:, 2] >= c.tilt_cos_min)
                  & (torch.linalg.vector_norm(state.angular_velocity, dim=1) <= c.angular_speed_tolerance)
                  & (torch.linalg.vector_norm(state.linear_velocity[:, :2] - commands.velocity[:, :2], dim=1)
                     <= c.horizontal_speed_tolerance))
        self.stable_time.copy_(torch.where(stable, self.stable_time + dt, 0.0))
        self.success_event.copy_(stable & (self.stable_time >= c.stable_landing_s))
        self.successes += self.success_event.float()
        self.exit_event.copy_(active & (self.success_event | (self.elapsed >= c.timeout_s)))
        self.phase[self.exit_event] = self.IDLE
        self.cooldown[self.exit_event] = c.cooldown_s
        self.elapsed[self.exit_event] = 0

        if c.automatic:
            self.requested |= ((self.phase == self.IDLE) & (self.cooldown <= 0)
                               & (torch.rand_like(self.elapsed) < min(c.rate_per_s * dt, 1.0)))
        ready = ((self.phase == self.IDLE) & (self.cooldown <= 0) & (episode_time >= c.min_episode_s)
                 & state.wheel_contact.all(dim=1) & state.finite & ~state.forbidden_contact
                 & ((state.height - commands.height).abs() <= c.height_tolerance)
                 & (state.linear_velocity[:, :2].norm(dim=1) <= c.horizontal_speed_tolerance)
                 & (-state.gravity[:, 2] >= c.tilt_cos_min))
        enter = self.requested & ready & ~self.exit_event
        if enter.any():
            self.requested[enter] = False
            self.trigger_event[enter] = True
            self.attempts[enter] += 1
            self.phase[enter] = self.PUSH
            self.elapsed[enter] = self.air_time[enter] = self.stable_time[enter] = 0
            self.max_upward_speed[enter] = 0
            self.seen_air[enter] = False
            self.start_height[enter] = state.height[enter]
            self.peak[enter] = state.height[enter]
            self.target_peak[enter] = torch.empty_like(self.target_peak[enter]).uniform_(*c.peak_height)
            self.release_velocity[enter] = (2 * 9.81 * (self.target_peak[enter] - c.release_height)).clamp(min=0).sqrt()
            push_time = 2 * (c.release_height - self.start_height[enter]).clamp(min=0.01) / self.release_velocity[enter].clamp(min=.1)
            self.push_duration[enter] = push_time.clamp(min=dt, max=c.max_push_s)

    def push_reference(self):
        t = torch.minimum(self.elapsed, self.push_duration)
        acceleration = self.release_velocity / self.push_duration.clamp(min=1e-6)
        return self.start_height + .5 * acceleration * t.square(), acceleration * t

    def observation(self):
        active = self.phase != self.IDLE
        # Same conceptual seven slots as the old task, but this is a NEW schema.
        flags = torch.zeros(len(self.phase), 5, device=self.phase.device)
        flags[:, 0] = (~active).float()
        flags[:, 4] = active.float()
        return torch.cat((flags, torch.where(active, self.target_peak, 0.0)[:, None],
                          torch.where(active, self.elapsed / self.cfg.timeout_s, 0.0)[:, None]), dim=1)
