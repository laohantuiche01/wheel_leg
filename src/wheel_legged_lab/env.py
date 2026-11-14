"""Single inheritance boundary. The environment only schedules its components."""
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import torch
import isaaclab.sim as sim
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from .config import WheelLeggedEnvCfg
from .robot import RobotDescription, RobotBinding
from .components.commands import CommandSampler
from .components.disturbances import MassRandomizer, ForceDisturbance
from .components.jump import JumpController
from .components.jump_task import MeasuredJumpController, jump_task_schema
from .components.motors import MotorBank, SpringController, AuxiliaryHoldController
from .components.observations import ObservationBuilder, observation_schema
from .components.reset import ResetController
from .components.rewards import RewardModel
from .components.state import StateReader
from .components.legacy_control import SustainedTermination
from .components.locomotion import LocomotionCommands, GimbalHeadingController

class WheelLeggedEnv(DirectRLEnv):
    cfg: WheelLeggedEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        self._components_ready = False
        super().__init__(cfg, render_mode, **kwargs)
        s = cfg.settings
        self.description = RobotDescription.load(s.robot_file)
        self.binding = RobotBinding(self.robot, self.contact_sensor, self.description)
        n, device = self.num_envs, self.device
        self.motors = MotorBank(self.robot, self.binding, self.description, s.motors, n, device, self.physics_dt)
        self.termination = SustainedTermination(s.termination, n, device)
        self.springs = SpringController(self.robot, self.binding, s.spring, n, device)
        self.auxiliary_holds = AuxiliaryHoldController(self.robot, self.binding, self.description, n, device)
        self.gimbal_heading = (GimbalHeadingController(self.robot, self.description, s.spin, n, device)
                               if s.spin.enabled else None)
        self.mass = MassRandomizer(self.robot, s.mass, n, device)
        self.force = ForceDisturbance(self.robot, s.force, n, device)
        command_type = LocomotionCommands if s.task in ("walk", "spin") else CommandSampler
        self.commands = command_type(s.commands, n, device)
        jump_type = MeasuredJumpController if s.jump.enabled and s.jump.implementation in ("measured_v1", "measured_v2") else JumpController
        self.jump = jump_type(s.jump, n, device)
        self.record_jump_transitions = False
        self.keyboard_control = None
        self.reader = StateReader(self.robot, self.contact_sensor, self.binding, self.description,
                                  self.scene.env_origins, s.contact_force_threshold, self.gimbal_heading)
        self.observations = ObservationBuilder(s.observations, self.robot, self.motors,
                                               self.commands, self.jump, self.mass, self.force, self.gimbal_heading)
        self.rewards = RewardModel(s.rewards, n, device, heading_sigma=s.spin.heading_sigma)
        self.reset_controller = ResetController(self.robot, self.scene.env_origins, s)
        self._all_ids = torch.arange(n, device=device)
        self._state_tick = self._reward_tick = -1
        self._components_ready = True
        self.mass.sample(self._all_ids)
        self.motors.reset(self._all_ids)
        self.springs.reset(self._all_ids)
        self.force.reset(self._all_ids)
        self.commands.reset(self._all_ids)
        if self.gimbal_heading is not None:
            self.gimbal_heading.reset(self._all_ids)
        self.observation_schema = observation_schema(self.description, s.observations, s.spin)
        self.observation_schema.update(
            asset_sha256=hashlib.sha256(Path(self.description.usd).read_bytes()).hexdigest(),
            observation_settings=asdict(s.observations),
            rest_joint_positions=self.description.rest_joint_positions,
            rest_height=self.description.rest_height, wheel_radius=self.description.wheel_radius,
            spring_settings=asdict(s.spring),
            auxiliary_holds={name: asdict(hold) for name, hold in self.description.auxiliary_holds.items()},
            motor_groups=[asdict(group) for group in self.description.motors])
        # Baseline defaults remain compatible with already validated v2 weights.
        from .settings import MotorSettings
        if asdict(s.motors) != asdict(MotorSettings()):
            self.observation_schema["motor_settings"] = asdict(s.motors)
        if s.task in ("walk", "spin"):
            self.observation_schema["motion_task"] = dict(task=s.task, commands=asdict(s.commands),
                spin=asdict(s.spin), velocity_reference=s.rewards.velocity_reference,
                support_all_ground=s.rewards.support_all_ground, lean_in_turns=s.rewards.lean_in_turns)
        if isinstance(self.jump, MeasuredJumpController):
            self.observation_schema["jump_task"] = jump_task_schema(s)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        sim.spawn_ground_plane("/World/ground", sim.GroundPlaneCfg(physics_material=self.cfg.sim.physics_material))
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        light = sim.DomeLightCfg(intensity=2000.0)
        light.func("/World/Light", light)

    def _pre_physics_step(self, actions):
        self.motors.accept(actions)
        self.force.advance(self.step_dt)

    def _apply_action(self):
        self.motors.apply()
        self.springs.apply()
        self.auxiliary_holds.apply()
        if self.gimbal_heading is not None:
            self.gimbal_heading.apply()
        self.force.apply()

    def _finish_step(self):
        # DirectRLEnv invokes dones before rewards; observations never advance time.
        if self._state_tick != self.common_step_counter:
            self._state = self.reader.read()
            self.jump.advance(self.step_dt, self._state, self.commands, self.episode_length_buf * self.step_dt)
            self._state_tick = self.common_step_counter
        return self._state

    def _get_dones(self):
        state = self._finish_step()
        tilt_limit = math.cos(math.radians(self.cfg.settings.termination_tilt_deg))
        cfg = self.cfg.settings.termination
        immediate = (~state.finite | state.forbidden_contact
            | (self.robot.data.joint_vel[:, self.binding.action_ids].abs().amax(dim=1) > cfg.joint_speed_limit)
            | (state.linear_velocity.abs().amax(dim=1) > cfg.root_linear_speed_limit)
            | (state.angular_velocity.abs().amax(dim=1) > cfg.root_angular_speed_limit))
        # Never let intermittent base/gimbal ground contact evade a consecutive
        # counter. Persistence is only for tilt, not forbidden-body collisions.
        ordinary = -state.gravity[:, 2] < tilt_limit
        terminate = self.termination.update(ordinary, immediate, self.common_step_counter).clone()
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        if isinstance(self.jump, MeasuredJumpController):
            self.jump.abort(terminate | timeout)
        return terminate, timeout

    def _get_rewards(self):
        if self._reward_tick != self.common_step_counter:
            self._reward = self.rewards.compute(self._finish_step(), self.commands, self.motors,
                                                self.jump, self.reset_terminated, self.step_dt)
            if self.record_jump_transitions and isinstance(self.jump, MeasuredJumpController):
                # Preserve the pre-reset terminal events for exact trial evaluation.
                self.extras["jump_transition"] = self.jump.transition_snapshot(self._state)
            # Commands for the NEXT policy decision, after the old command is scored.
            self.commands.advance(self.step_dt)
            self._reward_tick = self.common_step_counter
        return self._reward

    def _get_observations(self):
        return self.observations.build(self.reader.read(), self.common_step_counter)

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if self._components_ready:
            duration = self.episode_length_buf[env_ids] * self.step_dt
            logs = self.rewards.reset_logs(env_ids, duration)
            attempts = self.jump.attempts[env_ids].sum().clamp(min=1)
            logs["Episode/Jump/TakeoffPerAttempt"] = self.jump.takeoffs[env_ids].sum() / attempts
            logs["Episode/Jump/SuccessPerAttempt"] = self.jump.successes[env_ids].sum() / attempts
            logs["Episode/Jump/AttemptCount"] = self.jump.attempts[env_ids].mean()
            self.extras["log"] = logs
            if isinstance(self.jump, MeasuredJumpController):
                logs.update(self.jump.episode_logs(env_ids))
        super()._reset_idx(env_ids)
        if not self._components_ready:
            return
        self.reset_controller.apply(env_ids)
        self.motors.reset(env_ids)
        self.springs.reset(env_ids)
        self.auxiliary_holds.reset(env_ids)
        if self.gimbal_heading is not None:
            self.gimbal_heading.reset(env_ids)
        self.termination.reset(env_ids)
        self.observations.reset(env_ids)
        self.commands.reset(env_ids)
        self.jump.reset(env_ids)
        self.force.reset(env_ids)
        self.force.apply()  # clear simulator wrench buffers immediately
        if self.cfg.settings.mass.on_reset:
            self.mass.sample(env_ids)

    def _ids(self, env_ids):
        return self._all_ids if env_ids is None else torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

    def enable_keyboard_control(self, height_range=(.20, .38)):
        """Explicit playback override; call after validating/loading weights."""
        if self.keyboard_control is None:
            from .components.keyboard_mode import UnifiedKeyboardControl
            self.keyboard_control = UnifiedKeyboardControl(self, height_range)
        elif self.keyboard_control.height_range != tuple(height_range):
            raise ValueError("Keyboard control is already configured with another height range")
        return self.keyboard_control

    @torch.inference_mode()
    def set_play_velocity_command(self, vx=None, vy=None, wz=None, env_ids=None):
        if self.keyboard_control is not None:
            return self.keyboard_control.set_velocity(self._ids(env_ids), vx=vx, vy=vy, wz=wz)
        if self.cfg.settings.jump.enabled and any(value not in (None, 0, 0.0) for value in (vx, vy, wz)):
            raise ValueError("The initial jump task only supports in-place jumps")
        if self.cfg.settings.task == "spin" and (vx not in (None, 0, 0.0) or vy not in (None, 0, 0.0)):
            raise ValueError("Spin is in-place: translational commands must be zero")
        self.commands.set_manual(self._ids(env_ids), vx=vx, vy=vy, wz=wz)

    @torch.inference_mode()
    def set_gimbal_heading(self, heading, env_ids=None):
        if self.gimbal_heading is None:
            raise RuntimeError("World-frame gimbal heading control is only enabled for spin")
        self.gimbal_heading.set_target(self._ids(env_ids), heading)

    @torch.inference_mode()
    def set_height_command(self, height, env_ids=None):
        if self.keyboard_control is not None:
            return self.keyboard_control.set_height(self._ids(env_ids), height)
        if self.cfg.settings.jump.enabled and height >= self.cfg.settings.jump.release_height:
            raise ValueError("Jump standing height must be below release height")
        self.commands.set_manual(self._ids(env_ids), height=height)

    @torch.inference_mode()
    def request_jump_takeoff(self, env_ids=None):
        if self.keyboard_control is not None:
            return self.keyboard_control.request_jump(self._ids(env_ids))
        self.jump.request(self._ids(env_ids))

    @torch.inference_mode()
    def cancel_jump_request(self, env_ids=None):
        """Clear queued requests; an already active jump continues normally."""
        self.jump.requested[self._ids(env_ids)] = False

    @torch.inference_mode()
    def apply_external_force(self, force_xyz, duration_s, torque_xyz=(0., 0., 0.), env_ids=None):
        self.force.request(self._ids(env_ids), force_xyz, duration_s, torque_xyz)

    @torch.inference_mode()
    def set_mass_scale(self, scale, env_ids=None):
        self.mass.set_scale(self._ids(env_ids), scale)
