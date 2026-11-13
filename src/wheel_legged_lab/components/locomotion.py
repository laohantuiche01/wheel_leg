"""Slewed locomotion commands and a single owner for world-frame gimbal yaw."""
import math
import torch
from .commands import CommandSampler


def wrap_angle(value):
    return torch.atan2(torch.sin(value), torch.cos(value))


def heading_from_quaternion(quaternion):
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def world_to_heading(velocity, quaternion):
    heading = heading_from_quaternion(quaternion)
    c, s = heading.cos(), heading.sin()
    return torch.stack((c * velocity[:, 0] + s * velocity[:, 1],
                        -s * velocity[:, 0] + c * velocity[:, 1], velocity[:, 2]), dim=1)


class LocomotionCommands:
    """Reuse command sampling; ramp the public command at control frequency."""
    def __init__(self, cfg, n, device):
        self.cfg = cfg
        self.sampler = CommandSampler(cfg, n, device)
        self.velocity = torch.zeros(n, 3, device=device)
        self.height = self.sampler.height
        self.elapsed = torch.zeros(n, device=device)

    def reset(self, ids):
        self.sampler.reset(ids)
        self.velocity[ids] = 0
        self.elapsed[ids] = 0

    def advance(self, dt):
        self.sampler.advance(dt)
        self.elapsed += dt
        target = torch.where((self.elapsed >= self.cfg.settle_s)[:, None], self.sampler.velocity, 0.0)
        limits = torch.tensor([self.cfg.linear_accel, self.cfg.linear_accel, self.cfg.yaw_accel],
                              device=self.velocity.device) * dt
        self.velocity += (target - self.velocity).clamp(-limits, limits)

    def set_manual(self, ids, vx=None, vy=None, wz=None, height=None):
        if self.cfg.wz == (0.0, 0.0) and wz not in (None, 0, 0.0):
            raise ValueError("Translation-only commands require zero yaw rate")
        if self.cfg.vx == (0.0, 0.0) and vx not in (None, 0, 0.0):
            raise ValueError("In-place spin commands require zero forward velocity")
        self.sampler.set_manual(ids, vx=vx, vy=vy, wz=wz, height=height)

    def clear_manual(self, ids):
        self.sampler.clear_manual(ids)


class GimbalHeadingController:
    """Hold the yaw link's initial WORLD heading while the chassis rotates.

    The yaw axis is +Z in the v2 parent frame. The actuator must have zero
    stiffness/damping; this controller alone writes its feed-forward torque.
    """
    observation_width = 5

    def __init__(self, robot, spec, cfg, n, device):
        self.robot, self.cfg = robot, cfg
        if cfg.yaw_joint not in spec.auxiliary_joints or cfg.yaw_joint in spec.auxiliary_holds:
            raise ValueError("Gimbal yaw must be an auxiliary joint without another position hold")
        self.joint_id = robot.joint_names.index(cfg.yaw_joint)
        self.body_id = robot.body_names.index(cfg.yaw_body)
        self.target = torch.zeros(n, device=device)
        self.manual_target = torch.full_like(self.target, float("nan"))
        self.effort = torch.zeros(n, 1, device=device)

    def reset(self, ids):
        # Root pose was just written by ResetController. Avoid capturing stale
        # child-body poses before sim.forward() has propagated the new yaw.
        root_heading = heading_from_quaternion(self.robot.data.root_quat_w[ids])
        relative_yaw = self.robot.data.default_joint_pos[ids, self.joint_id]
        self.target[ids] = wrap_angle(root_heading + relative_yaw)
        manual = self.manual_target[ids]
        self.target[ids] = torch.where(torch.isfinite(manual), manual, self.target[ids])
        self.effort[ids] = 0

    def set_target(self, ids, heading):
        if not math.isfinite(float(heading)):
            raise ValueError("Gimbal heading must be finite radians")
        target = math.atan2(math.sin(heading), math.cos(heading))
        self.manual_target[ids] = target
        self.target[ids] = target

    def error(self):
        actual = heading_from_quaternion(self.robot.data.body_quat_w[:, self.body_id])
        return wrap_angle(self.target - actual)

    def angular_velocity(self):
        return self.robot.data.body_ang_vel_w[:, self.body_id, 2]

    def apply(self):
        effort = self.cfg.kp * self.error() - self.cfg.kd * self.angular_velocity()
        self.effort[:, 0] = effort.clamp(-self.cfg.effort_limit, self.cfg.effort_limit)
        self.robot.set_joint_effort_target(self.effort, joint_ids=[self.joint_id])

    def observation(self):
        q = self.robot.data.joint_pos[:, self.joint_id]
        qd = self.robot.data.joint_vel[:, self.joint_id]
        error = self.error()
        return torch.stack((q.sin(), q.cos(), qd * self.cfg.velocity_observation_scale,
                            error.sin(), error.cos()), dim=1)
