"""Explicit motor strategies + episode-wise latency/strength, composed by MotorBank."""
import torch
from .legacy_control import DifferenceVelocity, TorqueSpeedCurve

class PositionPD:
    def torque(self, action, q, qd, q0, group, limits):
        target = (q0 + action * group.action_scale).clamp(limits[..., 0], limits[..., 1])
        return group.kp * (target - q) - group.kd * qd

class VelocityPD:
    def torque(self, action, q, qd, q0, group, limits):
        target = (action * group.action_scale).clamp(-group.speed_limit, group.speed_limit)
        return group.kd * (target - qd)

class DirectTorque:
    def torque(self, action, q, qd, q0, group, limits):
        return action * group.action_scale

class MotorBank:
    def __init__(self, robot, binding, spec, cfg, n, device, physics_dt=0.005):
        self.robot, self.ids, self.cfg = robot, binding.action_ids, cfg
        a = len(self.ids)
        self.actions = torch.zeros(n, a, device=device)
        self.previous = torch.zeros_like(self.actions)
        self.before_previous = torch.zeros_like(self.actions)
        self.applied = torch.zeros_like(self.actions)
        self.effort = torch.zeros_like(self.actions)
        self.directions = torch.tensor([d for g in spec.motors for d in g.directions], device=device)
        self.strength = torch.ones_like(self.actions)
        self.delay = torch.zeros(n, dtype=torch.long, device=device)
        self.group_delay = torch.zeros(n, a, dtype=torch.long, device=device)
        self.history = torch.zeros(cfg.delay_steps[1] + 1, n, a, device=device)
        self.cursor = 0
        self.rows = torch.arange(n, device=device)
        self.groups = []
        start = 0
        for group in spec.motors:
            end = start + len(group.joints)
            strategy = {"position": PositionPD, "velocity": VelocityPD, "torque": DirectTorque}[group.mode]()
            self.groups.append((group, strategy, slice(start, end), self.ids[start:end]))
            start = end
        self.position_mask = torch.tensor([g.mode == "position" for g in spec.motors for _ in g.joints], device=device)
        self.velocity_estimator = DifferenceVelocity(n, a, device, cfg.velocity_filter_alpha)
        self.physics_dt = physics_dt
        self.curves = {name: TorqueSpeedCurve(points, device) for name, points in cfg.torque_speed_curves.items()}
        if set(self.curves) - {g.name for g in spec.motors}:
            raise ValueError("Torque curve refers to an unknown motor group")

    def reset(self, ids):
        for buf in (self.actions, self.previous, self.before_previous, self.applied, self.effort):
            buf[ids] = 0
        self.history[:, ids] = 0
        self.velocity_estimator.reset(ids)
        self.strength[ids] = torch.empty_like(self.strength[ids]).uniform_(*self.cfg.strength_range)
        self.delay[ids] = torch.randint(self.cfg.delay_steps[0], self.cfg.delay_steps[1] + 1,
                                        (len(ids),), device=self.delay.device)
        if self.cfg.delay_clock == "physics":
            for _, _, columns, _ in self.groups:
                self.group_delay[ids, columns] = torch.randint(self.cfg.delay_steps[0], self.cfg.delay_steps[1] + 1,
                    (len(ids), 1), device=self.delay.device)

    def accept(self, actions):
        if actions.shape != self.actions.shape or not torch.isfinite(actions).all():
            raise ValueError("Policy actions must have the configured shape and be finite")
        self.before_previous.copy_(self.previous)
        self.previous.copy_(self.actions)
        clipped = actions.clamp(-self.cfg.action_clip, self.cfg.action_clip)
        self.actions.copy_(self.cfg.action_filter_alpha * clipped + (1 - self.cfg.action_filter_alpha) * self.previous)
        if self.cfg.delay_clock == "control":
            self._advance_delay()

    def _advance_delay(self):
        self.history[self.cursor].copy_(self.actions)
        if self.cfg.delay_clock == "physics":
            columns = torch.arange(self.actions.shape[1], device=self.actions.device)
            self.applied.copy_(self.history[(self.cursor - self.group_delay) % len(self.history), self.rows[:, None], columns])
        else:
            self.applied.copy_(self.history[(self.cursor - self.delay) % len(self.history), self.rows])
        self.cursor = (self.cursor + 1) % len(self.history)

    def apply(self):
        data = self.robot.data
        if self.cfg.delay_clock == "physics":
            self._advance_delay()
        velocity = data.joint_vel[:, self.ids]
        if self.cfg.velocity_feedback == "finite_difference":
            velocity = self.velocity_estimator.sample(data.joint_pos[:, self.ids], velocity,
                self.physics_dt, ~self.position_mask)
        for group, strategy, columns, ids in self.groups:
            qd = velocity[:, columns]
            effort = strategy.torque(self.applied[:, columns] * self.directions[columns], data.joint_pos[:, ids], qd,
                data.default_joint_pos[:, ids], group, data.soft_joint_pos_limits[:, ids])
            effort = effort * self.strength[:, columns]
            limit = torch.full_like(effort, group.effort_limit)
            if group.name in self.curves:
                # Measured legacy envelope is symmetric, including braking.
                limit = torch.minimum(limit, self.curves[group.name].limit(data.joint_vel[:, ids]))
            if self.cfg.use_torque_speed_envelope:
                # Motoring torque falls with speed; braking retains nominal torque.
                motoring = effort * qd > 0
                envelope = (1.0 - qd.abs() / group.speed_limit).clamp(0.0, 1.0)
                limit = torch.where(motoring, limit * envelope, limit)
            effort = effort.clamp(-limit, limit)
            self.effort[:, columns] = effort * self.directions[columns]
            self.robot.set_joint_effort_target(effort, joint_ids=ids)

class AuxiliaryHoldController:
    """Fixed joint-relative PD targets, outside the six policy action channels."""
    def __init__(self, robot, binding, spec, n, device):
        self.robot, self.ids = robot, binding.hold_ids
        holds = list(spec.auxiliary_holds.values())
        def values(name):
            return torch.tensor([getattr(hold, name) for hold in holds], dtype=torch.float32, device=device)
        self.target = values("target")
        self.kp, self.kd, self.limit = values("kp"), values("kd"), values("effort_limit")
        self.effort = torch.zeros(n, len(self.ids), device=device)
        limits = robot.data.soft_joint_pos_limits[:, self.ids]
        if ((self.target < limits[..., 0]) | (self.target > limits[..., 1])).any():
            raise ValueError("Auxiliary hold target is outside joint soft limits")

    def reset(self, ids):
        self.effort[ids] = 0

    def apply(self):
        if not self.ids:
            return
        data = self.robot.data
        effort = self.kp * (self.target - data.joint_pos[:, self.ids]) - self.kd * data.joint_vel[:, self.ids]
        self.effort.copy_(effort.clamp(-self.limit, self.limit))
        self.robot.set_joint_effort_target(self.effort, joint_ids=self.ids)


class SpringController:
    def __init__(self, robot, binding, cfg, n, device):
        self.robot, self.ids, self.cfg = robot, binding.spring_ids, cfg
        self.offset = torch.zeros(n, len(self.ids), device=device)
        self.reference = (robot.data.default_joint_pos[:, self.ids].clone() if cfg.offset is None
                          else torch.full_like(self.offset, cfg.offset))

    def reset(self, ids):
        self.offset[ids] = torch.empty_like(self.offset[ids]).uniform_(*self.cfg.noise_range)

    def apply(self):
        if not self.ids:
            return
        qd = self.robot.data.joint_vel[:, self.ids]
        force = -self.cfg.damping * qd
        if self.cfg.enabled:
            compression = (self.reference - self.robot.data.joint_pos[:, self.ids]).clamp(min=0.0)
            force = force + self.cfg.preload + self.cfg.stiffness * compression + self.offset
        self.robot.set_joint_effort_target(force.clamp(-self.cfg.effort_limit, self.cfg.effort_limit), joint_ids=self.ids)
