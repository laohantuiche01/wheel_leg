"""Asset description and name resolution. No references to V13/V14 environments."""
from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import re

@dataclass(frozen=True)
class MotorGroup:
    name: str
    joints: tuple[str, ...]
    mode: str
    action_scale: float
    kp: float
    kd: float
    effort_limit: float
    speed_limit: float
    armature: float
    directions: tuple[float, ...] = ()  # policy coordinate -> physical joint coordinate

@dataclass(frozen=True)
class PositionHold:
    target: float
    kp: float
    kd: float
    effort_limit: float
    speed_limit: float

@dataclass
class RobotDescription:
    name: str
    usd: str
    base_body: str
    wheel_bodies: list[str]
    termination_bodies: list[str]
    rest_height: float
    wheel_radius: float
    rest_joint_positions: dict[str, float]
    motors: list[MotorGroup]
    passive_joint_patterns: list[str]
    spring_joints: list[str]
    auxiliary_joints: list[str]
    joint_names: list[str] = field(default_factory=list)
    auxiliary_holds: dict[str, PositionHold] = field(default_factory=dict)

    @classmethod
    def load(cls, filename):
        path = Path(filename)
        data = json.loads(path.read_text())
        data["usd"] = str((path.parent / data["usd"]).resolve())
        data["motors"] = [MotorGroup(**{**g, "joints": tuple(g["joints"]),
            "directions": tuple(g.get("directions", [1.0] * len(g["joints"])))}) for g in data["motors"]]
        data["auxiliary_holds"] = {name: PositionHold(**cfg) for name, cfg in data.get("auxiliary_holds", {}).items()}
        spec = cls(**data)
        names = spec.action_names
        if not names or len(names) != len(set(names)):
            raise ValueError("Every policy joint must occur exactly once")
        if len(spec.wheel_bodies) != 2:
            raise ValueError("The current balance/jump components require exactly two wheels")
        if not Path(spec.usd).is_file():
            raise FileNotFoundError(spec.usd)
        group_names = [g.name for g in spec.motors]
        if (len(group_names) != len(set(group_names)) or set(group_names) & {"passive", "springs", "auxiliary", "gimbal_heading"}
                or any(name.startswith("auxiliary_hold_") for name in group_names)):
            raise ValueError("Motor group names must be unique and not reserved")
        for group in spec.motors:
            if group.mode not in ("position", "velocity", "torque") or min(group.effort_limit, group.speed_limit) <= 0:
                raise ValueError(f"Invalid motor group: {group.name}")
            if len(group.directions) != len(group.joints) or any(d not in (-1, 1) for d in group.directions):
                raise ValueError(f"Motor directions must contain one +1/-1 per joint: {group.name}")
        if spec.joint_names:
            spec.validate_joints(spec.joint_names)
        if not set(spec.auxiliary_holds) <= set(spec.auxiliary_joints):
            raise ValueError("Position holds may only own auxiliary joints")
        for name, hold in spec.auxiliary_holds.items():
            if (not all(math.isfinite(value) for value in vars(hold).values())
                    or min(hold.kp, hold.effort_limit, hold.speed_limit) <= 0 or hold.kd < 0):
                raise ValueError(f"Invalid auxiliary hold: {name}")
        return spec

    @property
    def action_names(self):
        return [joint for group in self.motors for joint in group.joints]

    def validate_joints(self, names):
        """Reject omissions, ambiguous resets and stale names before control starts."""
        if len(names) != len(set(names)):
            raise ValueError("Articulation joint names must be unique")
        if self.joint_names and set(names) != set(self.joint_names):
            raise ValueError(f"Articulation joint inventory mismatch: {sorted(set(names) ^ set(self.joint_names))}")
        owners = self.action_names + self.spring_joints + self.auxiliary_joints
        for pattern in self.passive_joint_patterns:
            matches = [name for name in names if re.fullmatch(pattern, name)]
            if not matches:
                raise ValueError(f"Passive joint pattern matches nothing: {pattern}")
            owners.extend(matches)
        if len(owners) != len(set(owners)) or set(owners) != set(names):
            raise ValueError("Every articulation joint must have exactly one motor/passive owner")
        reset_names = []
        for pattern in self.rest_joint_positions:
            matches = [name for name in names if re.fullmatch(pattern, name)]
            if not matches:
                raise ValueError(f"Rest pose pattern matches nothing: {pattern}")
            reset_names.extend(matches)
        if len(reset_names) != len(set(reset_names)) or set(reset_names) != set(names):
            raise ValueError("Rest pose must cover every articulation joint exactly once")

    def build_articulation(self, spin=None):
        import isaaclab.sim as sim
        from isaaclab.assets import ArticulationCfg
        from isaaclab.actuators import IdealPDActuatorCfg
        # Policy PD is computed in MotorBank at physics frequency. The actuator
        # accepts effort only, preventing duplicate PD gains in two places.
        actuators = {
            g.name: IdealPDActuatorCfg(joint_names_expr=list(g.joints), stiffness=0.0, damping=0.0,
                effort_limit=g.effort_limit, effort_limit_sim=g.effort_limit,
                velocity_limit=g.speed_limit, velocity_limit_sim=g.speed_limit, armature=g.armature)
            for g in self.motors
        }
        if self.passive_joint_patterns:
            actuators["passive"] = IdealPDActuatorCfg(joint_names_expr=self.passive_joint_patterns,
                stiffness=0.0, damping=0.01, effort_limit=50.0, velocity_limit=300.0, armature=0.0001)
        if self.spring_joints:
            actuators["springs"] = IdealPDActuatorCfg(joint_names_expr=self.spring_joints,
                stiffness=0.0, damping=0.0, effort_limit=1000.0, velocity_limit=50.0, armature=0.0001)
        heading_joint = spin.yaw_joint if spin is not None and spin.enabled else None
        if heading_joint is not None:
            if heading_joint not in self.auxiliary_joints or heading_joint in self.auxiliary_holds:
                raise ValueError("Gimbal heading control must have exclusive auxiliary joint ownership")
            actuators["gimbal_heading"] = IdealPDActuatorCfg(joint_names_expr=[heading_joint],
                stiffness=0.0, damping=0.0, effort_limit=spin.effort_limit, effort_limit_sim=spin.effort_limit,
                velocity_limit=spin.speed_limit, velocity_limit_sim=spin.speed_limit, armature=0.0001)
        damped = [name for name in self.auxiliary_joints if name not in self.auxiliary_holds and name != heading_joint]
        if damped:
            actuators["auxiliary"] = IdealPDActuatorCfg(joint_names_expr=damped,
                stiffness=0.0, damping=0.5, effort_limit=2.0, velocity_limit=30.0, armature=0.0001)
        for name, hold in self.auxiliary_holds.items():
            actuators[f"auxiliary_hold_{name}"] = IdealPDActuatorCfg(joint_names_expr=[name],
                stiffness=0.0, damping=0.0, effort_limit=hold.effort_limit, effort_limit_sim=hold.effort_limit,
                velocity_limit=hold.speed_limit, velocity_limit_sim=hold.speed_limit, armature=0.0001)
        return ArticulationCfg(prim_path="/World/envs/env_.*/Robot",
            spawn=sim.UsdFileCfg(usd_path=self.usd, activate_contact_sensors=True,
                rigid_props=sim.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
                articulation_props=sim.ArticulationRootPropertiesCfg(enabled_self_collisions=False,
                    solver_position_iteration_count=12, solver_velocity_iteration_count=6)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, self.rest_height),
                joint_pos=self.rest_joint_positions, joint_vel={".*": 0.0}), actuators=actuators)

class RobotBinding:
    """Resolve articulation and sensor indices independently, once at startup."""
    def __init__(self, robot, contact_sensor, spec):
        def exact(names, available):
            missing = set(names) - set(available)
            if missing:
                raise ValueError(f"Asset names not found: {sorted(missing)}")
            return [available.index(name) for name in names]
        spec.validate_joints(robot.joint_names)
        self.base_body_id = exact([spec.base_body], robot.body_names)[0]
        self.action_ids = exact(spec.action_names, robot.joint_names)
        self.spring_ids = exact(spec.spring_joints, robot.joint_names)
        self.hold_ids = exact(list(spec.auxiliary_holds), robot.joint_names)
        self.wheel_body_ids = exact(spec.wheel_bodies, robot.body_names)
        self.wheel_sensor_ids = exact(spec.wheel_bodies, contact_sensor.body_names)
        self.termination_sensor_ids = exact(spec.termination_bodies, contact_sensor.body_names)
        auxiliary = exact(spec.auxiliary_joints, robot.joint_names)
        passive = [i for i, name in enumerate(robot.joint_names)
                   if any(re.fullmatch(pattern, name) for pattern in spec.passive_joint_patterns)]
        assigned = self.action_ids + self.spring_ids + auxiliary + passive
        if len(set(assigned)) != len(assigned):
            raise ValueError("Policy, spring, auxiliary and passive joint ownership must not overlap")
        if set(assigned) != set(range(len(robot.joint_names))):
            raise ValueError("Every articulation joint must have an explicit motor/passive owner")
