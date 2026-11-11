"""Versioned observation contract. Never silently load a shape-compatible legacy policy."""
import torch

SCHEMA_VERSION = "wheel-legged-composition-v2"

def observation_schema(spec):
    a = len(spec.action_names)
    return dict(version=SCHEMA_VERSION, robot=spec.name, action_names=spec.action_names,
        motor_modes=[g.mode for g in spec.motors], action_scales=[g.action_scale for g in spec.motors],
        action_directions=[d for g in spec.motors for d in g.directions],
        actor_dim=17 + 3*a, critic_dim=30 + 5*a,
        actor_blocks=["command_xyz", "height_command", "angular_velocity", "gravity",
                      "joint_position_offset", "joint_velocity", "actions", "task_mode_7"],
        critic_extras=["linear_velocity", "height", "wheel_contacts_2", "motor_effort",
                       "mass_scale", "force_xyz", "torque_xyz", "motor_strength"])

class ObservationBuilder:
    def __init__(self, cfg, robot, motors, commands, jump, mass, force):
        self.cfg, self.robot, self.motors = cfg, robot, motors
        self.commands, self.jump, self.mass, self.force = commands, jump, mass, force

    def _actor(self, state, noisy):
        c, d, m = self.cfg, self.robot.data, self.motors
        q = d.joint_pos[:, m.ids] - d.default_joint_pos[:, m.ids]
        q = q * m.position_mask * m.directions
        qd = d.joint_vel[:, m.ids] * m.directions
        angular, gravity = state.angular_velocity, state.gravity
        if noisy and c.noise_enabled:
            def noise(x, scale):
                return x + (2 * torch.rand_like(x) - 1) * scale
            q = noise(q, c.joint_position_noise) * m.position_mask
            qd = noise(qd, c.joint_velocity_noise)
            angular = noise(angular, c.angular_velocity_noise)
            gravity = noise(gravity, c.gravity_noise)
        return torch.cat((self.commands.velocity, self.commands.height[:, None],
                          angular * c.angular_velocity_scale, gravity, q,
                          qd * c.joint_velocity_scale, m.actions, self.jump.observation()), dim=1)

    def build(self, state):
        actor = self._actor(state, noisy=True)
        critic = torch.cat((self._actor(state, noisy=False), state.linear_velocity, state.height[:, None],
                            state.wheel_contact.float(), self.motors.effort,
                            self.mass.scale, self.force.force[:, 0], self.force.torque[:, 0],
                            self.motors.strength), dim=1)
        # Non-finite physical state is terminated separately, never rewarded.
        return {"policy": torch.nan_to_num(actor).clamp(-self.cfg.clip, self.cfg.clip),
                "critic": torch.nan_to_num(critic).clamp(-self.cfg.clip, self.cfg.clip)}
