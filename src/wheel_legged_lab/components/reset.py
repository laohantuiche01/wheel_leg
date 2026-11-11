"""Restore ALL joints to the asset's calibrated rest pose; no inherited IK."""
import torch
from isaaclab.utils.math import quat_from_euler_xyz

class ResetController:
    def __init__(self, robot, origins, cfg):
        self.robot, self.origins, self.cfg = robot, origins, cfg

    def apply(self, ids):
        root = self.robot.data.default_root_state[ids].clone()
        root[:, :3] += self.origins[ids]
        yaw = torch.empty(len(ids), device=root.device).uniform_(*self.cfg.reset_yaw_range)
        zeros = torch.zeros_like(yaw)
        root[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
        root[:, 7:] = 0
        q = self.robot.data.default_joint_pos[ids].clone()
        qd = torch.zeros_like(q)
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids=ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids=ids)
        self.robot.write_joint_state_to_sim(q, qd, env_ids=ids)
        self.robot.set_joint_effort_target(torch.zeros_like(q), env_ids=ids)
