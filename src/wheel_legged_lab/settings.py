"""Typed, simulator-independent settings. All units are SI / output-shaft units."""
from dataclasses import dataclass, field
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent / "assets"

@dataclass
class MotorSettings:
    action_clip: float = 10.0
    delay_steps: tuple[int, int] = (0, 0)  # sampled per episode, CONTROL steps
    strength_range: tuple[float, float] = (1.0, 1.0)
    use_torque_speed_envelope: bool = False

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

@dataclass
class JumpSettings:
    enabled: bool = False
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

@dataclass
class RewardSettings:
    weights: dict[str, float] = field(default_factory=lambda: {
        "velocity": 2.0, "velocity_error": -0.25,
        "yaw": 1.5, "yaw_error": -0.25,
        "height": 1.0, "height_error": -20.0,
        "upright": 1.0, "vertical_velocity": -0.5,
        "roll_error": -20.0, "pitch_error": -10.0, "roll_pitch_rate": -0.2,
        "torque": -0.0001, "action_rate": -0.01,
        "termination": -10.0,
        "jump_push_velocity": 4.0, "jump_push_height": 2.0,
        "jump_airtime": 3.0, "jump_peak": 50.0, "jump_success": 100.0,
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
    observations: ObservationSettings = field(default_factory=ObservationSettings)
    rewards: RewardSettings = field(default_factory=RewardSettings)
    reset_yaw_range: tuple[float, float] = (-3.14159265, 3.14159265)
    termination_tilt_deg: float = 40.0
    contact_force_threshold: float = 5.0

    def validate(self):
        ranges = (self.motors.delay_steps, self.motors.strength_range, self.mass.scale_range,
                  self.force.interval_s, self.force.duration_s, self.commands.vx,
                  self.commands.wz, self.commands.height, self.commands.interval_s,
                  self.jump.peak_height, self.reset_yaw_range, self.spring.noise_range)
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

def settings_for(task: str, robot_file: str | None = None) -> TaskSettings:
    from .robot import RobotDescription
    s = TaskSettings(task=task)
    if robot_file is not None:
        s.robot_file = str(Path(robot_file).resolve())
    spec = RobotDescription.load(s.robot_file)
    if task not in ("stand", "height", "velocity", "jump"):
        raise ValueError(f"Unknown task: {task}")
    if task != "velocity":
        s.commands.vx = (0.0, 0.0)
        s.commands.wz = (0.0, 0.0)
        s.commands.standing_fraction = 1.0
        s.commands.straight_fraction = s.commands.spin_fraction = 0.0
    if task == "stand":
        s.commands.height = (spec.rest_height, spec.rest_height)
    if task == "jump":
        s.commands.height = (0.28, 0.28)
        s.jump.enabled = True
    s.validate()
    return s
