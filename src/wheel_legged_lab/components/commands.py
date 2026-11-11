"""Body-frame velocity and standing-height commands, with explicit manual override."""
import math
import torch

class CommandSampler:
    def __init__(self, cfg, n, device):
        self.cfg = cfg
        self.velocity = torch.zeros(n, 3, device=device)
        self.height = torch.zeros(n, device=device)
        self.remaining = torch.zeros(n, device=device)
        self.manual_velocity = torch.full_like(self.velocity, float("nan"))
        self.manual_height = torch.full_like(self.height, float("nan"))

    def reset(self, ids):
        c = self.cfg
        count = len(ids)
        v = torch.zeros(count, 3, device=self.velocity.device)
        v[:, 0].uniform_(*c.vx)
        v[:, 2].uniform_(*c.wz)
        standing = torch.rand(count, device=v.device) < c.standing_fraction
        bucket = torch.rand(count, device=v.device)
        straight = ~standing & (bucket < c.straight_fraction)
        spin = ~standing & (bucket >= c.straight_fraction) & (bucket < c.straight_fraction + c.spin_fraction)
        v[standing] = 0
        v[straight, 2] = 0
        v[spin, 0] = 0
        if spin.any():
            signs = torch.where(torch.rand(int(spin.sum()), device=v.device) < 0.5, -1., 1.)
            upper = torch.where(signs < 0, -c.wz[0], c.wz[1])
            v[spin, 2] = signs * (c.spin_min_speed + torch.rand_like(signs) * (upper - c.spin_min_speed))
        self.velocity[ids] = v
        self.height[ids] = torch.empty(count, device=v.device).uniform_(*c.height)
        self.remaining[ids] = torch.empty(count, device=v.device).uniform_(*c.interval_s)
        self._apply_manual()

    def advance(self, dt):
        self.remaining.sub_(dt)
        ids = torch.nonzero(self.remaining <= 0, as_tuple=False).flatten()
        if len(ids):
            self.reset(ids)
        self._apply_manual()

    def set_manual(self, ids, vx=None, vy=None, wz=None, height=None):
        if any(value is not None and not math.isfinite(float(value)) for value in (vx, vy, wz, height)):
            raise ValueError("Manual commands must be finite")
        # Lateral translation is not an objective of a two-wheel base.
        if vy not in (None, 0, 0.0):
            raise ValueError("This task supports vx/wz commands, not lateral velocity")
        for axis, value in enumerate((vx, vy, wz)):
            if value is not None:
                self.manual_velocity[ids, axis] = float(value)
        if height is not None:
            if height <= 0:
                raise ValueError("Height must be positive")
            self.manual_height[ids] = height
        self._apply_manual()

    def clear_manual(self, ids):
        self.manual_velocity[ids] = float("nan")
        self.manual_height[ids] = float("nan")
        self.reset(ids)

    def _apply_manual(self):
        self.velocity.copy_(torch.where(torch.isfinite(self.manual_velocity), self.manual_velocity, self.velocity))
        self.height.copy_(torch.where(torch.isfinite(self.manual_height), self.manual_height, self.height))
