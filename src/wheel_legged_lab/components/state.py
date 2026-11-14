"""One measured snapshot shared by termination, jump and reward computation."""
from dataclasses import dataclass
import torch
from .locomotion import world_to_heading

def body_tilt_angles(gravity):
    """Signed roll/pitch in radians from body-frame projected gravity."""
    roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
    pitch = torch.atan2(gravity[:, 0], gravity[:, 1:].norm(dim=1))
    return roll, pitch

def wheel_fore_aft_difference(root_quat, wheel_positions):
    """Signed left-minus-right wheel-center offset along the body's X axis."""
    vector = wheel_positions[:, 0] - wheel_positions[:, 1]
    imaginary = root_quat[:, 1:]
    cross = torch.linalg.cross(imaginary, vector, dim=-1)
    body_vector = vector - 2 * root_quat[:, :1] * cross + 2 * torch.linalg.cross(imaginary, cross, dim=-1)
    return body_vector[:, 0]

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
    wheel_fore_aft_difference: torch.Tensor | None = None
    non_wheel_contact: torch.Tensor | None = None
    axle_linear_velocity_heading: torch.Tensor | None = None
    axle_position_world: torch.Tensor | None = None
    yaw_rate_world: torch.Tensor | None = None
    gimbal_heading_error: torch.Tensor | None = None
    gimbal_world_yaw_rate: torch.Tensor | None = None
    wheel_contact_force: torch.Tensor | None = None

class StateReader:
    def __init__(self, robot, sensor, binding, spec, origins, threshold, heading=None):
        self.robot, self.sensor, self.binding = robot, sensor, binding
        self.spec, self.origins, self.threshold = spec, origins, threshold
        self.non_wheel_ids = [i for i in range(len(sensor.body_names)) if i not in binding.wheel_sensor_ids]
        self.heading = heading

    def read(self):
        d, b = self.robot.data, self.binding
        force = torch.linalg.vector_norm(self.sensor.data.net_forces_w, dim=-1)
        # The CAD base origin is ~17 cm ahead of the axle. During in-place
        # spinning that origin necessarily orbits; track the axle, not that orbit.
        body_velocity = (d.body_link_lin_vel_w if hasattr(d, "body_link_lin_vel_w") else d.body_lin_vel_w)
        axle_velocity = body_velocity[:, b.wheel_body_ids].mean(dim=1)
        finite = (torch.isfinite(d.root_state_w).all(dim=1)
                  & torch.isfinite(d.joint_pos).all(dim=1) & torch.isfinite(d.joint_vel).all(dim=1)
                  & torch.isfinite(force).all(dim=1) & torch.isfinite(axle_velocity).all(dim=1))
        heading_error = self.heading.error() if self.heading is not None else None
        heading_rate = self.heading.angular_velocity() if self.heading is not None else None
        if heading_error is not None:
            finite &= torch.isfinite(heading_error) & torch.isfinite(heading_rate)
        return RobotState(
            height=d.root_pos_w[:, 2] - self.origins[:, 2],
            linear_velocity=d.root_lin_vel_b, angular_velocity=d.root_ang_vel_b,
            vertical_velocity=d.root_lin_vel_w[:, 2], gravity=d.projected_gravity_b,
            wheel_clearance=d.body_pos_w[:, b.wheel_body_ids, 2] - self.origins[:, 2:3] - self.spec.wheel_radius,
            wheel_contact=force[:, b.wheel_sensor_ids] > self.threshold,
            forbidden_contact=(force[:, b.termination_sensor_ids] > self.threshold).any(dim=1),
            finite=finite,
            wheel_fore_aft_difference=wheel_fore_aft_difference(d.root_quat_w, d.body_pos_w[:, b.wheel_body_ids]),
            non_wheel_contact=(force[:, self.non_wheel_ids] > self.threshold).any(dim=1),
            axle_linear_velocity_heading=world_to_heading(axle_velocity, d.root_quat_w),
            axle_position_world=d.body_pos_w[:, b.wheel_body_ids].mean(dim=1),
            yaw_rate_world=d.root_ang_vel_w[:, 2],
            gimbal_heading_error=heading_error, gimbal_world_yaw_rate=heading_rate,
            wheel_contact_force=force[:, b.wheel_sensor_ids])
