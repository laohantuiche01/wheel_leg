"""Mass/inertia randomization and one owner for external world-frame wrenches."""
import torch

class MassRandomizer:
    def __init__(self, robot, cfg, n, device):
        self.robot, self.cfg = robot, cfg
        self.ids = [robot.body_names.index(name) for name in cfg.bodies]
        self.base_mass = robot.root_physx_view.get_masses().clone().cpu()
        self.base_inertia = robot.root_physx_view.get_inertias().clone().cpu()
        self.scale = torch.ones(n, 1, device=device)

    def sample(self, env_ids):
        if self.cfg.enabled:
            scale = torch.empty(len(env_ids), 1).uniform_(*self.cfg.scale_range)
            self.set_scale(env_ids, scale)

    def set_scale(self, env_ids, scale):
        cpu_ids = env_ids.detach().cpu()
        scale = torch.as_tensor(scale, dtype=self.base_mass.dtype, device="cpu").reshape(-1, 1)
        if scale.shape[0] not in (1, len(cpu_ids)) or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Mass scales must be positive finite values, scalar or one per environment")
        # Always scale the immutable nominal values, never accumulate previous draws.
        masses = self.robot.root_physx_view.get_masses().clone()
        inertia = self.robot.root_physx_view.get_inertias().clone()
        masses[cpu_ids[:, None], self.ids] = self.base_mass[cpu_ids[:, None], self.ids] * scale
        inertia[cpu_ids[:, None], self.ids] = self.base_inertia[cpu_ids[:, None], self.ids] * scale.unsqueeze(-1)
        self.robot.root_physx_view.set_masses(masses, cpu_ids)
        self.robot.root_physx_view.set_inertias(inertia, cpu_ids)
        self.scale[env_ids] = scale.to(self.scale.device)

class ForceDisturbance:
    def __init__(self, robot, cfg, n, device):
        self.robot, self.cfg = robot, cfg
        self.body_id = robot.body_names.index(cfg.body)
        self.force = torch.zeros(n, 1, 3, device=device)
        self.torque = torch.zeros_like(self.force)
        self.remaining = torch.zeros(n, device=device)
        self.wait = torch.zeros_like(self.remaining)

    def reset(self, ids):
        self.force[ids] = 0
        self.torque[ids] = 0
        self.remaining[ids] = 0
        self.wait[ids] = torch.empty_like(self.wait[ids]).uniform_(*self.cfg.interval_s)

    def request(self, ids, force_xyz, duration_s, torque_xyz=(0.0, 0.0, 0.0)):
        if duration_s <= 0:
            raise ValueError("Force duration must be positive")
        force = torch.as_tensor(force_xyz, dtype=self.force.dtype, device=self.force.device)
        torque = torch.as_tensor(torque_xyz, dtype=self.torque.dtype, device=self.torque.device)
        if force.shape != (3,) or torque.shape != (3,) or not torch.isfinite(force).all() or not torch.isfinite(torque).all():
            raise ValueError("Force and torque must be finite world-frame xyz vectors")
        self.force[ids, 0] = force
        self.torque[ids, 0] = torque
        self.remaining[ids] = duration_s

    def advance(self, dt):
        expired = self.remaining <= 0
        self.force[expired] = 0
        self.torque[expired] = 0
        self.wait.sub_(dt)
        if self.cfg.enabled:
            ids = torch.nonzero((self.wait <= 0) & expired, as_tuple=False).flatten()
            if len(ids):
                fmax = torch.tensor(self.cfg.force_xyz, device=self.force.device)
                tmax = torch.tensor(self.cfg.torque_xyz, device=self.force.device)
                self.force[ids, 0] = (2 * torch.rand(len(ids), 3, device=self.force.device) - 1) * fmax
                self.torque[ids, 0] = (2 * torch.rand(len(ids), 3, device=self.force.device) - 1) * tmax
                self.remaining[ids] = torch.empty_like(self.remaining[ids]).uniform_(*self.cfg.duration_s)
                self.wait[ids] = self.remaining[ids] + torch.empty_like(self.wait[ids]).uniform_(*self.cfg.interval_s)
        self.remaining.sub_(dt).clamp_(min=0.0)

    def apply(self):
        # Explicit zero buffers also clear expired pushes in PhysX.
        self.robot.set_external_force_and_torque(self.force, self.torque, body_ids=[self.body_id], is_global=True)
