"""One measured snapshot shared by termination, jump and reward computation."""
from dataclasses import dataclass
import torch

def body_tilt_angles(gravity):
    """Signed roll/pitch in radians from body-frame projected gravity."""
    roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
    pitch = torch.atan2(gravity[:, 0], gravity[:, 1:].norm(dim=1))
    return roll, pitch

@dataclass
class RobotState:
    height: torch.Tensor
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor
    vertical_velocity: torch.Tensor
    gravity: torch.Tensor
    wheel_clearance: torch.Tensor
    wheel_contact: torch.Tensor
    forbidden_contact: torch.Tensor
    finite: torch.Tensor

class StateReader:
    def __init__(self, robot, sensor, binding, spec, origins, threshold):
        self.robot, self.sensor, self.binding = robot, sensor, binding
        self.spec, self.origins, self.threshold = spec, origins, threshold

    def read(self):
        d, b = self.robot.data, self.binding
        force = torch.linalg.vector_norm(self.sensor.data.net_forces_w, dim=-1)
        finite = (torch.isfinite(d.root_state_w).all(dim=1)
                  & torch.isfinite(d.joint_pos).all(dim=1) & torch.isfinite(d.joint_vel).all(dim=1)
                  & torch.isfinite(force).all(dim=1))
        return RobotState(
            height=d.root_pos_w[:, 2] - self.origins[:, 2],
            linear_velocity=d.root_lin_vel_b, angular_velocity=d.root_ang_vel_b,
            vertical_velocity=d.root_lin_vel_w[:, 2], gravity=d.projected_gravity_b,
            wheel_clearance=d.body_pos_w[:, b.wheel_body_ids, 2] - self.origins[:, 2:3] - self.spec.wheel_radius,
            wheel_contact=force[:, b.wheel_sensor_ids] > self.threshold,
            forbidden_contact=(force[:, b.termination_sensor_ids] > self.threshold).any(dim=1),
            finite=finite)
