"""Auditable reward terms and actual-episode-duration logging."""
import torch
from .state import body_tilt_angles

class RewardModel:
    def __init__(self, cfg, n, device, heading_sigma=0.2):
        self.weights = dict(cfg.weights)
        self.cfg, self.heading_sigma = cfg, heading_sigma
        self.sums = {name: torch.zeros(n, device=device) for name in self.weights}
        self.metric_sums = {name: torch.zeros(n, device=device) for name in
            ("RollRMSDeg", "PitchRMSDeg", "TiltRMSDeg", "WheelHeightDiffRMSM", "WheelStaggerRMSM", "BothWheelsContactFraction",
             "VelocityErrorRMS", "YawRateErrorRMS", "LateralSpeedRMS", "GimbalHeadingRMSDeg", "GimbalWorldYawRateRMS")}
        self.metric_time = torch.zeros(n, device=device)

    def compute(self, state, commands, motors, jump, terminated, dt):
        linear_velocity, yaw_rate = state.linear_velocity, state.angular_velocity[:, 2]
        if getattr(self.cfg, "velocity_reference", "base_body") == "axle_heading":
            if state.axle_linear_velocity_heading is None or state.yaw_rate_world is None:
                raise ValueError("Axle velocity and world yaw rate are required for locomotion rewards")
            linear_velocity, yaw_rate = state.axle_linear_velocity_heading, state.yaw_rate_world
        vx_error = commands.velocity[:, 0] - linear_velocity[:, 0]
        yaw_error = commands.velocity[:, 2] - yaw_rate
        height_error = commands.height - state.height
        ground_phase = ((jump.phase == jump.IDLE) | (jump.phase == jump.LAND)).float()
        reference_height, reference_velocity = jump.push_reference()
        push = (jump.phase == jump.PUSH).float()
        upright = (-state.gravity[:, 2]).clamp(0., 1.)
        roll, pitch = body_tilt_angles(state.gravity)
        stagger = getattr(state, "wheel_fore_aft_difference", None)
        stagger = torch.zeros_like(state.height) if stagger is None else stagger
        standing = ground_phase * (commands.velocity.abs().amax(dim=1) < 0.1).float()
        support = ground_phase if getattr(self.cfg, "support_all_ground", False) else standing
        reference_roll = (-(commands.velocity[:, 0] * commands.velocity[:, 2] / 9.81).atan().clamp(-.2, .2)
                          if getattr(self.cfg, "lean_in_turns", False) else 0.0)
        heading_error = getattr(state, "gimbal_heading_error", None)
        heading_error = torch.zeros_like(state.height) if heading_error is None else heading_error
        heading_rate = getattr(state, "gimbal_world_yaw_rate", None)
        heading_rate = torch.zeros_like(state.height) if heading_rate is None else heading_rate
        non_wheel = getattr(state, "non_wheel_contact", None)
        non_wheel = torch.zeros_like(state.height) if non_wheel is None else non_wheel.float()
        second_diff = motors.actions - 2 * motors.previous + getattr(motors, "before_previous", motors.previous)
        if hasattr(motors, "robot"):
            velocity = motors.robot.data.joint_vel[:, motors.ids] * motors.directions
            wheel_power = (motors.effort * velocity)[:, ~motors.position_mask].clamp(min=0).sum(dim=1)
        else:
            wheel_power = torch.zeros_like(state.height)
        terms = {
            "velocity": torch.exp(-(vx_error / getattr(self.cfg, "velocity_sigma", 1.0)).square()),
            "velocity_error": vx_error.square(),
            "yaw": torch.exp(-yaw_error.square() / .5), "yaw_error": yaw_error.square(),
            "height": ground_phase * torch.exp(-height_error.square() / .01),
            "height_error": ground_phase * height_error.square(), "upright": upright.square(),
            "roll_error": ground_phase * (roll - reference_roll).square(),
            "pitch_error": ground_phase * pitch.square(),
            "roll_pitch_rate": ground_phase * state.angular_velocity[:, :2].square().sum(dim=1),
            "wheel_stagger": support * stagger.square(),
            "double_support": support * state.wheel_contact.all(dim=1).float(),
            "standing_wheel_clearance": support * state.wheel_clearance.clamp(min=0).square().sum(dim=1),
            "non_wheel_contact": non_wheel,
            "lateral_velocity": linear_velocity[:, 1].square(),
            "spin_drift": linear_velocity[:, :2].square().sum(dim=1),
            "gimbal_heading": torch.exp(-(heading_error / self.heading_sigma).square()),
            "gimbal_heading_error": heading_error.square(),
            "vertical_velocity": ground_phase * state.vertical_velocity.square(),
            "torque": motors.effort.square().sum(dim=1),
            "action_rate": (motors.actions - motors.previous).square().sum(dim=1),
            "action_smoothness": second_diff.square().sum(dim=1), "wheel_power": wheel_power,
            "termination": terminated.float(),
            "jump_push_velocity": push * torch.exp(-(state.vertical_velocity - reference_velocity).square() / .5),
            "jump_push_height": push * torch.exp(-(state.height - reference_height).square() / .003),
            "jump_airtime": (jump.air_now & jump.seen_air).float(),
            "jump_peak": (jump.exit_event & jump.seen_air).float() * torch.exp(-(jump.peak - jump.target_peak).square() / .003),
            "jump_success": jump.success_event.float(),
        }
        missing = set(self.weights) - set(terms)
        if missing:
            raise ValueError(f"Configured reward terms not implemented: {sorted(missing)}")
        total = torch.zeros_like(state.height)
        for name, weight in self.weights.items():
            contribution = torch.nan_to_num(terms[name] * weight * dt)
            contribution = torch.where(state.finite, contribution, torch.zeros_like(contribution))
            # Numerical failures still receive an explicit terminal penalty.
            if name == "termination":
                contribution = terminated.float() * weight * dt
            self.sums[name] += contribution
            total += contribution
        metrics = dict(RollRMSDeg=roll.rad2deg().square(), PitchRMSDeg=pitch.rad2deg().square(),
            TiltRMSDeg=torch.acos((-state.gravity[:, 2]).clamp(-1, 1)).rad2deg().square(),
            WheelHeightDiffRMSM=(state.wheel_clearance[:, 0] - state.wheel_clearance[:, 1]).square(),
            WheelStaggerRMSM=stagger.square(),
            VelocityErrorRMS=vx_error.square(), YawRateErrorRMS=yaw_error.square(),
            LateralSpeedRMS=linear_velocity[:, 1].square(), GimbalHeadingRMSDeg=heading_error.rad2deg().square(),
            GimbalWorldYawRateRMS=heading_rate.square(),
            BothWheelsContactFraction=state.wheel_contact.all(dim=1).float())
        for name, value in metrics.items():
            self.metric_sums[name] += torch.where(state.finite, torch.nan_to_num(value), 0.0) * dt
        self.metric_time += state.finite.float() * dt
        return total

    def reset_logs(self, ids, duration):
        logs = {f"Episode/Reward/{name}": (value[ids] / duration.clamp(min=1e-6)).mean()
                for name, value in self.sums.items()}
        for value in self.sums.values():
            value[ids] = 0
        # Use actual measured time, independent of randomized initial episode ages.
        elapsed = self.metric_time[ids].clamp(min=1e-6)
        for name, value in self.metric_sums.items():
            mean = value[ids] / elapsed
            logs[f"Episode/Posture/{name}"] = (mean.sqrt() if "RMS" in name else mean).mean()
            value[ids] = 0
        self.metric_time[ids] = 0
        return logs
