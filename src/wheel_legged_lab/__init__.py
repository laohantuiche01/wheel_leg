"""Explicit registration; importing this package does not import old project tasks."""
import gymnasium as gym

for _task in ("stand", "height", "velocity", "jump"):
    gym.register(id=f"WheelLegged-{_task.title()}-Direct-v0",
        entry_point="wheel_legged_lab.env:WheelLeggedEnv", disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"wheel_legged_lab.config:{_task}_env_cfg",
                "rsl_rl_cfg_entry_point": "wheel_legged_lab.ppo:WheelLeggedPPORunnerCfg"})
