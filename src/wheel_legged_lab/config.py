"""One Isaac Lab configuration class; task differences are data, not subclasses."""
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from .settings import TaskSettings, settings_for
from .robot import RobotDescription
from .components.observations import observation_schema

@configclass
class WheelLeggedEnvCfg(DirectRLEnvCfg):
    settings: TaskSettings = TaskSettings()
    decimation = 4
    episode_length_s = 20.0
    action_space = 6
    observation_space = 35
    state_space = 60
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=0.005, render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0,
                                                restitution=0.0, friction_combine_mode="multiply"))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=4.0, replicate_physics=True)
    contact_sensor: ContactSensorCfg = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*",
                                                      update_period=0.0, history_length=3)
    robot_cfg = None

    def __post_init__(self):
        self.settings.validate()
        spec = RobotDescription.load(self.settings.robot_file)
        self.robot_cfg = spec.build_articulation()
        schema = observation_schema(spec)
        self.action_space = len(spec.action_names)
        self.observation_space = schema["actor_dim"]
        self.state_space = schema["critic_dim"]

def stand_env_cfg():
    return WheelLeggedEnvCfg(settings=settings_for("stand"))

def height_env_cfg():
    return WheelLeggedEnvCfg(settings=settings_for("height"))

def velocity_env_cfg():
    return WheelLeggedEnvCfg(settings=settings_for("velocity"))

def jump_env_cfg():
    return WheelLeggedEnvCfg(settings=settings_for("jump"))
