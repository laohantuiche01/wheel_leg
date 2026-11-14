"""Typed, simulator-independent settings. All units are SI / output-shaft units."""
from dataclasses import dataclass, field
from pathlib import Path
import math

ASSET_DIR = Path(__file__).resolve().parent / "assets"
TASKS = ("stand", "height", "velocity", "jump", "walk", "spin")

@dataclass
class MotorSettings:
    action_clip: float = 10.0
    delay_steps: tuple[int, int] = (0, 0)  # sampled per episode, CONTROL steps
    delay_clock: str = "control"
    strength_range: tuple[float, float] = (1.0, 1.0)
    use_torque_speed_envelope: bool = False
    action_filter_alpha: float = 1.0  # new action weight; old optional EMA used 0.8
    velocity_feedback: str = "simulator"  # or finite_difference, sampled at physics rate
    velocity_filter_alpha: float = 1.0
    torque_speed_curves: dict[str, list[list[float]]] = field(default_factory=dict)  # output rad/s, Nm

@dataclass
class TerminationSettings:
    duration_steps: int = 1  # tilt only; forbidden-body contacts always terminate immediately
    joint_speed_limit: float = 500.0
    root_linear_speed_limit: float = 100.0
    root_angular_speed_limit: float = 200.0

@dataclass
class MassSettings:
    enabled: bool = False
    bodies: tuple[str, ...] = ("base_link",)
    scale_range: tuple[float, float] = (0.9, 1.1)
    on_reset: bool = False  # startup only by default (PhysX CPU property API)

@dataclass
class ForceSettings:
    enabled: bool = False
    body: str = "base_link"
    interval_s: tuple[float, float] = (5.0, 10.0)
    duration_s: tuple[float, float] = (0.1, 0.3)
    force_xyz: tuple[float, float, float] = (30.0, 30.0, 0.0)
    torque_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)

@dataclass
class SpringSettings:
    enabled: bool = True
    offset: float | None = None  # None: each spring's model-specific reset coordinate
    preload: float = 400.0
    stiffness: float = (600.0 - 400.0) / 0.07
    damping: float = 50.0
    effort_limit: float = 1000.0
    noise_range: tuple[float, float] = (0.0, 0.0)

@dataclass
class CommandSettings:
    vx: tuple[float, float] = (-3.0, 3.0)
    wz: tuple[float, float] = (-2.0, 2.0)
    height: tuple[float, float] = (0.20, 0.34)
    interval_s: tuple[float, float] = (3.0, 8.0)
    standing_fraction: float = 0.15
    straight_fraction: float = 0.25  # fractions of NON-standing commands
    spin_fraction: float = 0.25
    spin_min_speed: float = 0.5
    settle_s: float = 1.0
    linear_accel: float = 0.8
    yaw_accel: float = 1.5

@dataclass
class SpinSettings:
    enabled: bool = False
    yaw_joint: str = "gimbal_yaw_joint"
    yaw_body: str = "gimbal_yaw_link"
    kp: float = 20.0
    kd: float = 1.5
    effort_limit: float = 2.0
    speed_limit: float = 30.0
    velocity_observation_scale: float = 0.1
    heading_sigma: float = 0.2  # radians

@dataclass
class JumpSettings:
    enabled: bool = False
    implementation: str = "legacy"
    automatic: bool = True
    rate_per_s: float = 0.5
    min_episode_s: float = 2.0
    cooldown_s: float = 2.0
    peak_height: tuple[float, float] = (0.40, 0.44)
    release_height: float = 0.34
    max_push_s: float = 0.30
    timeout_s: float = 1.5
    min_air_s: float = 0.04
    air_clearance: float = 0.02
    min_upward_speed: float = 0.3
    stable_landing_s: float = 0.2
    height_tolerance: float = 0.06
    horizontal_speed_tolerance: float = 0.4
    vertical_speed_tolerance: float = 0.3
    tilt_cos_min: float = 0.94
    angular_speed_tolerance: float = 1.5
    release_speed_fraction: float = 0.6
    release_force_max: float = 20.0
    push_timeout_s: float = 0.28

@dataclass
class ObservationSettings:
    noise_enabled: bool = True
    angular_velocity_noise: float = 0.05
    gravity_noise: float = 0.02
    joint_position_noise: float = 0.01
    joint_velocity_noise: float = 0.1
    angular_velocity_scale: float = 0.25
    joint_velocity_scale: float = 0.05
    clip: float = 100.0
    actor_history: int = 1
    actor_delay: int = 0  # control frames; separate from history length

@dataclass
class RewardSettings:
    velocity_sigma: float = 1.0  # m/s; original exp(-error**2) at the default
    velocity_reference: str = "base_body"
    support_all_ground: bool = False
    lean_in_turns: bool = False
    push_regularization_scale: float = 1.0
    weights: dict[str, float] = field(default_factory=lambda: {
        "velocity": 2.0, "velocity_error": -0.25,
        "yaw": 1.5, "yaw_error": -0.25,
        "height": 1.0, "height_error": -80.0,
        "upright": 1.0, "vertical_velocity": -0.5,
        "roll_error": -20.0, "pitch_error": -10.0, "roll_pitch_rate": -0.2,
        "wheel_stagger": -50.0, "double_support": 2.0,
        "standing_wheel_clearance": -80.0, "non_wheel_contact": -2.0,
        "lateral_velocity": 0.0, "spin_drift": 0.0,
        "gimbal_heading": 0.0, "gimbal_heading_error": 0.0,
        "torque": -0.0001, "action_rate": -0.01,
        "action_smoothness": 0.0, "wheel_power": 0.0,
        "termination": -10.0,
        "jump_push_velocity": 4.0, "jump_push_height": 2.0,
        "jump_airtime": 3.0, "jump_peak": 50.0, "jump_success": 100.0,
        "jump_takeoff": 0.0, "jump_failed": 0.0,
        "jump_push_height_progress": 0.0,
    })

@dataclass
class TaskSettings:
    robot_file: str = str(ASSET_DIR / "v2_0.json")
    task: str = "velocity"
    motors: MotorSettings = field(default_factory=MotorSettings)
    mass: MassSettings = field(default_factory=MassSettings)
    force: ForceSettings = field(default_factory=ForceSettings)
    spring: SpringSettings = field(default_factory=SpringSettings)
    commands: CommandSettings = field(default_factory=CommandSettings)
    jump: JumpSettings = field(default_factory=JumpSettings)
    spin: SpinSettings = field(default_factory=SpinSettings)
    observations: ObservationSettings = field(default_factory=ObservationSettings)
    rewards: RewardSettings = field(default_factory=RewardSettings)
    termination: TerminationSettings = field(default_factory=TerminationSettings)
    reset_yaw_range: tuple[float, float] = (-3.14159265, 3.14159265)
    reset_linear_velocity_range: tuple[float, float] = (0.0, 0.0)  # world xyz, m/s
    reset_angular_velocity_range: tuple[float, float] = (0.0, 0.0)  # world xyz, rad/s
    termination_tilt_deg: float = 40.0
    contact_force_threshold: float = 5.0

    def validate(self):
        if not math.isfinite(self.rewards.velocity_sigma) or self.rewards.velocity_sigma <= 0:
            raise ValueError("Velocity reward sigma must be finite and positive")
        if self.task not in TASKS:
            raise ValueError(f"Unknown task: {self.task}")
        if self.rewards.velocity_reference not in ("base_body", "axle_heading"):
            raise ValueError("Unknown velocity reference")
        if self.commands.settle_s < 0 or min(self.commands.linear_accel, self.commands.yaw_accel) <= 0:
            raise ValueError("Invalid locomotion command ramp")
        if self.spin.enabled and (self.task != "spin" or self.jump.enabled):
            raise ValueError("World-heading control is reserved for the spin task")
        if self.task == "spin" and (not self.spin.enabled or self.commands.vx != (0.0, 0.0)):
            raise ValueError("Spin requires heading control and zero translational command")
        if min(self.spin.kp, self.spin.effort_limit, self.spin.speed_limit, self.spin.heading_sigma) <= 0 or self.spin.kd < 0:
            raise ValueError("Invalid gimbal heading controller")
        if not 0 < self.motors.action_filter_alpha <= 1 or not 0 < self.motors.velocity_filter_alpha <= 1:
            raise ValueError("Filter coefficients must lie in (0, 1]")
        if self.motors.velocity_feedback not in ("simulator", "finite_difference"):
            raise ValueError("Unsupported motor velocity feedback")
        if self.motors.delay_clock not in ("control", "physics"):
            raise ValueError("Unsupported motor delay clock")
        if int(self.observations.actor_delay) != self.observations.actor_delay or self.observations.actor_delay < 0:
            raise ValueError("Observation delay must be a nonnegative integer")
        for count in (self.observations.actor_history, self.termination.duration_steps):
            if isinstance(count, bool) or int(count) != count or count < 1:
                raise ValueError("History and termination lengths must be positive integers")
        if min(self.termination.joint_speed_limit, self.termination.root_linear_speed_limit,
               self.termination.root_angular_speed_limit) <= 0:
            raise ValueError("Termination speed limits must be positive")
        ranges = (self.motors.delay_steps, self.motors.strength_range, self.mass.scale_range,
                  self.force.interval_s, self.force.duration_s, self.commands.vx,
                  self.commands.wz, self.commands.height, self.commands.interval_s,
                  self.jump.peak_height, self.reset_yaw_range, self.spring.noise_range,
                  self.reset_linear_velocity_range, self.reset_angular_velocity_range)
        if any(len(r) != 2 or r[0] > r[1] for r in ranges):
            raise ValueError("Ranges must be ordered pairs")
        if self.motors.delay_steps[0] < 0 or any(int(x) != x for x in self.motors.delay_steps):
            raise ValueError("Motor delays must be nonnegative integer control steps")
        if min(self.mass.scale_range) <= 0 or min(self.motors.strength_range) <= 0:
            raise ValueError("Mass and motor strengths must remain positive")
        c = self.commands
        if not 0 <= c.standing_fraction <= 1 or min(c.straight_fraction, c.spin_fraction) < 0 or c.straight_fraction + c.spin_fraction > 1:
            raise ValueError("Invalid command mixture probabilities")
        if min(c.interval_s) <= 0 or min(self.force.interval_s) <= 0 or min(self.force.duration_s) <= 0:
            raise ValueError("Intervals and durations must be positive")
        if c.spin_fraction > 0 and (min(abs(c.wz[0]), abs(c.wz[1])) < c.spin_min_speed or not c.wz[0] < 0 < c.wz[1]):
            raise ValueError("Spin commands require positive and negative yaw ranges above spin_min_speed")
        if self.jump.enabled and (c.vx != (0.0, 0.0) or c.wz != (0.0, 0.0)):
            raise ValueError("First jump task supports in-place jumps only")
        if self.jump.enabled and (self.jump.peak_height[0] <= self.jump.release_height or self.jump.release_height <= c.height[1]):
            raise ValueError("Require standing height < release height < peak height")
        if self.jump.timeout_s <= self.jump.max_push_s + self.jump.stable_landing_s:
            raise ValueError("Jump timeout must leave time for landing")
        j = self.jump
        if j.implementation not in ("legacy", "measured_v1", "measured_v2"):
            raise ValueError("Unknown jump implementation")
        if not math.isfinite(self.rewards.push_regularization_scale) or not 0 < self.rewards.push_regularization_scale <= 1:
            raise ValueError("Push regularization scale must lie in (0, 1]")
        jump_values = (j.rate_per_s, j.min_episode_s, j.cooldown_s, *j.peak_height, j.release_height,
            j.max_push_s, j.timeout_s, j.min_air_s, j.air_clearance, j.min_upward_speed,
            j.stable_landing_s, j.height_tolerance, j.horizontal_speed_tolerance,
            j.vertical_speed_tolerance, j.angular_speed_tolerance, j.release_speed_fraction,
            j.release_force_max, j.push_timeout_s, j.tilt_cos_min)
        if not all(math.isfinite(v) for v in jump_values) or min(jump_values) < 0:
            raise ValueError("Jump parameters must be finite and nonnegative")
        if (min(j.max_push_s, j.min_air_s, j.min_upward_speed, j.stable_landing_s,
                j.height_tolerance, j.vertical_speed_tolerance, j.angular_speed_tolerance) <= 0
                or not 0 < j.release_speed_fraction <= 1 or not 0 < j.tilt_cos_min <= 1
                or not 0 < j.push_timeout_s < j.timeout_s):
            raise ValueError("Invalid jump timing or release/landing gates")

def apply_jump_reward_profile(settings):
    """Versioned profiles keep old jump checkpoint evaluation reproducible."""
    dense = settings.jump.implementation == "measured_v2"
    settings.rewards.push_regularization_scale = .2 if dense else 1.0
    settings.rewards.weights.update(jump_push_velocity=10.0 if dense else 1.0, jump_push_height=.25,
        jump_push_height_progress=4.0 if dense else 0.0, jump_takeoff=200.0 if dense else 100.0,
        jump_success=300.0 if dense else 200.0, jump_failed=-50.0 if dense else -25.0)


def settings_for(task: str, robot_file: str | None = None, strategy: str = "baseline") -> TaskSettings:
    from .robot import RobotDescription
    s = TaskSettings(task=task)
    if robot_file is not None:
        s.robot_file = str(Path(robot_file).resolve())
    spec = RobotDescription.load(s.robot_file)
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    if task not in ("velocity", "walk", "spin"):
        s.commands.vx = (0.0, 0.0)
        s.commands.wz = (0.0, 0.0)
        s.commands.standing_fraction = 1.0
        s.commands.straight_fraction = s.commands.spin_fraction = 0.0
    if task == "stand":
        s.commands.height = (spec.rest_height, spec.rest_height)
    if task == "jump":
        s.commands.height = (0.28, 0.28)
        s.jump.enabled = True
        s.jump.implementation = "measured_v2"
        apply_jump_reward_profile(s)
    if task in ("walk", "spin"):
        s.commands.height = (0.35, 0.35)
        s.rewards.velocity_reference = "axle_heading"
        s.rewards.support_all_ground = True
        s.rewards.weights.update(lateral_velocity=-1.0, wheel_stagger=-20.0,
                                 standing_wheel_clearance=-80.0, action_smoothness=-0.001)
    if task == "walk":
        s.commands.vx, s.commands.wz = (-2.0, 2.0), (0.0, 0.0)
        s.commands.interval_s = (4.0, 8.0)
        s.commands.standing_fraction = 0.1
        s.commands.straight_fraction, s.commands.spin_fraction = 1.0, 0.0
        s.rewards.lean_in_turns = False
        s.rewards.weights.update(velocity=3.0, velocity_error=-1.0)
    if task == "spin":
        s.spin.enabled = True
        s.commands.vx, s.commands.wz = (0.0, 0.0), (-2.0, 2.0)
        s.commands.interval_s = (30.0, 40.0)  # keep the selected direction within a 20s episode
        s.commands.standing_fraction = s.commands.straight_fraction = 0.0
        s.commands.spin_fraction, s.commands.spin_min_speed = 1.0, 1.0
        s.rewards.weights.update(yaw=3.0, yaw_error=-1.0, spin_drift=-3.0,
                                 gimbal_heading=1.0, gimbal_heading_error=-1.0)
    if strategy == "legacy_ppo":
        s.motors.action_filter_alpha = 0.8
        s.motors.velocity_feedback = "finite_difference"
        s.motors.velocity_filter_alpha = 0.2
        s.termination.duration_steps = 20
        s.motors.delay_steps = (1, 3)
        s.motors.delay_clock = "physics"
        s.observations.actor_delay = 1
    elif strategy != "baseline":
        raise ValueError(f"Unknown strategy: {strategy}")
    s.validate()
    return s
