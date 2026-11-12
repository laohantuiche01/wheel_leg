"""Simulator-independent control mechanisms adapted from the old PPO project.

Source: wheeled-legged_RL action filter, DiffVelPDActuator, M3508Actuator,
and termination-duration logic. No imports of old task classes.
"""
import csv
import math
import torch


class DifferenceVelocity:
    def __init__(self, n, a, device, alpha=1.0):
        self.previous = torch.zeros(n, a, device=device)
        self.filtered = torch.zeros_like(self.previous)
        self.valid = torch.zeros(n, dtype=torch.bool, device=device)
        self.alpha = alpha

    def reset(self, ids):
        self.valid[ids] = False
        self.filtered[ids] = 0

    def sample(self, q, simulator_velocity, dt, wrap_mask):
        if dt <= 0:
            raise ValueError("Velocity sampling dt must be positive")
        delta = q - self.previous
        wrapped = torch.atan2(torch.sin(delta), torch.cos(delta))
        delta = torch.where(wrap_mask, wrapped, delta)
        raw = delta / dt
        # Explicit reset() handles teleports. Large unexpected jumps retain the
        # old estimator's fallback (1 rad over one sample, per environment).
        valid = self.valid & (delta.abs().amax(dim=1) <= 1.0)
        estimate = self.alpha * raw + (1 - self.alpha) * self.filtered
        self.filtered.copy_(torch.where(valid[:, None], estimate, simulator_velocity))
        self.previous.copy_(q)
        self.valid.fill_(True)
        return self.filtered


class TorqueSpeedCurve:
    def __init__(self, points, device):
        if (len(points) < 2 or any(len(p) != 2 or not all(math.isfinite(x) for x in p) for p in points)
                or any(p[0] < 0 or p[1] < 0 for p in points)
                or any(a[0] >= b[0] for a, b in zip(points, points[1:]))):
            raise ValueError("Torque curves need finite, nonnegative points with strictly increasing speed")
        self.speed = torch.tensor([p[0] for p in points], dtype=torch.float32, device=device)
        self.torque = torch.tensor([p[1] for p in points], dtype=torch.float32, device=device)

    @staticmethod
    def read_csv(filename, *, gear_ratio, efficiency=1.0):
        """Old measured CSV motor-shaft rpm/Nm -> output-shaft rad/s/Nm."""
        if not math.isfinite(gear_ratio) or gear_ratio <= 0 or not 0 < efficiency <= 1:
            raise ValueError("Invalid drivetrain conversion")
        with open(filename, newline="") as stream:
            rows = list(csv.DictReader(stream))
        points = [(abs(float(r["Motor_shaft_speed_rpm"])) * 2 * math.pi / (60 * gear_ratio),
                   abs(float(r["Motor_shaft_torque_Nm_ideal"])) * gear_ratio * efficiency) for r in rows]
        # Consolidate repeated measured speeds using the conservative torque.
        unique = {}
        for speed, torque in points:
            unique[speed] = min(unique.get(speed, torque), torque)
        return [list(pair) for pair in sorted(unique.items())]

    def limit(self, velocity):
        speed = velocity.abs()
        hi = torch.searchsorted(self.speed, speed.contiguous()).clamp(1, len(self.speed) - 1)
        lo = hi - 1
        alpha = (speed - self.speed[lo]) / (self.speed[hi] - self.speed[lo])
        result = self.torque[lo] + alpha * (self.torque[hi] - self.torque[lo])
        result = torch.where(speed <= self.speed[0], self.torque[0], result)
        return torch.where(speed > self.speed[-1], 0.0, result).clamp(min=0)


class SustainedTermination:
    def __init__(self, cfg, n, device):
        self.cfg = cfg
        self.count = torch.zeros(n, dtype=torch.long, device=device)
        self.last_tick = None
        self.result = torch.zeros(n, dtype=torch.bool, device=device)

    def reset(self, ids):
        self.count[ids] = 0
        self.result[ids] = False

    def update(self, ordinary, immediate, tick):
        if self.last_tick != tick:
            self.count.copy_(torch.where(ordinary, self.count + 1, 0))
            self.result.copy_(immediate | (self.count >= self.cfg.duration_steps))
            self.last_tick = tick
        return self.result


class ObservationHistory:
    """Oldest -> newest; repeated reads replace the current frame, never shift."""
    def __init__(self, n, length, width, device):
        self.frames = torch.zeros(n, length, width, device=device)
        self.last_tick = None

    def reset(self, ids):
        self.frames[ids] = 0

    def update(self, frame, tick):
        if tick != self.last_tick:
            self.frames[:, :-1] = self.frames[:, 1:].clone()
            self.last_tick = tick
        self.frames[:, -1] = frame
        return self.frames.flatten(1).clone()
