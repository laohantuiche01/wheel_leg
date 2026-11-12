"""Independent port of the old ordinary-PPO exploration-noise floor."""
import math
import torch
from rsl_rl.runners import OnPolicyRunner


def clamp_policy_noise(policy, floor):
    if not math.isfinite(floor) or floor < 0:
        raise ValueError("Noise floor must be finite and nonnegative")
    if floor == 0:
        return
    with torch.no_grad():
        std = getattr(policy, "std", None)
        log_std = getattr(policy, "log_std", None)
        if isinstance(std, torch.nn.Parameter):
            std.clamp_(min=floor)
        if isinstance(log_std, torch.nn.Parameter):
            log_std.clamp_(min=math.log(floor))


class NoiseFloorRunner(OnPolicyRunner):
    """Clamp after initialization/load and each logged PPO update, as in legacy.

    The training entry point always supplies a log directory. Inference uses
    standard OnPolicyRunner and does not change the Gaussian mean or weights.
    """
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.noise_std_floor = train_cfg.get("noise_std_floor", 0.1)
        if log_dir is None:
            raise ValueError("NoiseFloorRunner training requires a log directory")
        super().__init__(env, train_cfg, log_dir, device)
        clamp_policy_noise(self.alg.policy, self.noise_std_floor)

    def load(self, *args, **kwargs):
        result = super().load(*args, **kwargs)
        clamp_policy_noise(self.alg.policy, self.noise_std_floor)
        return result

    def log(self, locs, width=80, pad=35):
        clamp_policy_noise(self.alg.policy, self.noise_std_floor)
        return super().log(locs, width=width, pad=pad)
