"""Auditable reward terms and actual-episode-duration logging."""
import torch
from .state import body_tilt_angles

class RewardModel:
    def __init__(self, cfg, n, device):
        self.weights = dict(cfg.weights)
        self.sums = {name: torch.zeros(n, device=device) for name in self.weights}
        self.metric_sums = {name: torch.zeros(n, device=device) for name in
            ("RollRMSDeg", "PitchRMSDeg", "TiltRMSDeg", "WheelHeightDiffRMSM", "BothWheelsContactFraction")}
        self.metric_time = torch.zeros(n, device=device)

    def compute(self, state, commands, motors, jump, terminated, dt):
        vx_error = commands.velocity[:, 0] - state.linear_velocity[:, 0]
        yaw_error = commands.velocity[:, 2] - state.angular_velocity[:, 2]
        height_error = commands.height - state.height
        ground_phase = ((jump.phase == jump.IDLE) | (jump.phase == jump.LAND)).float()
        reference_height, reference_velocity = jump.push_reference()
        push = (jump.phase == jump.PUSH).float()
        upright = (-state.gravity[:, 2]).clamp(0., 1.)
        roll, pitch = body_tilt_angles(state.gravity)
        second_diff = motors.actions - 2 * motors.previous + getattr(motors, "before_previous", motors.previous)
        if hasattr(motors, "robot"):
            velocity = motors.robot.data.joint_vel[:, motors.ids] * motors.directions
            wheel_power = (motors.effort * velocity)[:, ~motors.position_mask].clamp(min=0).sum(dim=1)
        else:
            wheel_power = torch.zeros_like(state.height)
        terms = {
            "velocity": torch.exp(-vx_error.square()), "velocity_error": vx_error.square(),
            "yaw": torch.exp(-yaw_error.square() / .5), "yaw_error": yaw_error.square(),
            "height": ground_phase * torch.exp(-height_error.square() / .01),
            "height_error": ground_phase * height_error.square(), "upright": upright.square(),
            "roll_error": ground_phase * roll.square(),
            "pitch_error": ground_phase * pitch.square(),
            "roll_pitch_rate": ground_phase * state.angular_velocity[:, :2].square().sum(dim=1),
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
