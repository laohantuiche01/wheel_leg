"""One PPO configuration independent of old runner subclasses."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class WheelLeggedPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "wheel_legged_composition"
    policy = RslRlPpoActorCriticCfg(init_noise_std=0.25, actor_hidden_dims=[256, 128, 64],
                                   critic_hidden_dims=[256, 128, 64], activation="elu")
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=4.0, use_clipped_value_loss=True,
        clip_param=0.2, entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=1e-4, schedule="fixed", gamma=0.99, lam=0.95, desired_kl=None, max_grad_norm=1.0)
