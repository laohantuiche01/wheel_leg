"""Resolve playback configuration from checkpoint metadata before opening Kit."""
import json
from pathlib import Path


def resolve_playback_args(args):
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    args.checkpoint = checkpoint
    schema_path = checkpoint.parent / "observation_schema.json"
    if not schema_path.is_file():
        raise ValueError(f"Checkpoint observation schema is missing: {schema_path}")
    schema = json.loads(schema_path.read_text())
    launch_path = checkpoint.parent / "launch.json"
    launch = json.loads(launch_path.read_text()) if launch_path.is_file() else {}
    saved_strategy = launch.get("strategy", "baseline")
    saved_history = schema.get("observation_settings", {}).get("actor_history", 1)
    for field, saved in (("strategy", saved_strategy), ("actor_history", saved_history)):
        explicit = getattr(args, field, None)
        if explicit is not None and explicit != saved:
            raise ValueError(f"Playback --{field}={explicit!r} differs from training ({saved!r}). "
                             f"Use --{field} {saved}, or omit it to restore the saved configuration.")
        setattr(args, field, saved)
    if args.strategy not in ("baseline", "legacy_ppo") or not isinstance(args.actor_history, int) or args.actor_history < 1:
        raise ValueError("Invalid saved playback strategy/history")
    saved_task = "jump" if schema.get("jump_task") else launch.get("task", schema.get("motion_task", {}).get("task", "stand"))
    if (schema.get("motion_task") or schema.get("jump_task")) and args.task is not None and args.task != saved_task:
        raise ValueError(f"Checkpoint task is {saved_task}; use the matching task or omit --task")
    args.task = args.task or saved_task
    if args.task not in ("stand", "height", "velocity", "jump", "walk", "spin"):
        raise ValueError("Invalid saved task")
    args.seed = args.seed if args.seed is not None else launch.get("seed", 42)
    if args.robot_spec is None:
        args.robot_spec = launch.get("robot_spec")
    return schema


def restore_runtime_settings(settings, schema):
    """Keep trained controller/noise scales even if profile defaults later change."""
    for group, key in (("motors", "motor_settings"), ("spring", "spring_settings"),
                       ("observations", "observation_settings")):
        target = getattr(settings, group)
        for name, value in schema.get(key, {}).items():
            if not hasattr(target, name):
                raise ValueError(f"Unsupported saved setting: {group}.{name}")
            if isinstance(getattr(target, name), tuple):
                value = tuple(value)
            setattr(target, name, value)
    motion = schema.get("motion_task", {})
    for group in ("commands", "spin"):
        target = getattr(settings, group)
        for name, value in motion.get(group, {}).items():
            if not hasattr(target, name):
                raise ValueError(f"Unsupported saved motion setting: {group}.{name}")
            setattr(target, name, tuple(value) if isinstance(getattr(target, name), tuple) else value)
    for key in ("velocity_reference", "support_all_ground", "lean_in_turns"):
        if key in motion:
            setattr(settings.rewards, key, motion[key])
    if settings.task == "jump":
        restore_jump_settings(settings, schema)
    settings.observations.noise_enabled = False
    settings.validate()


def restore_jump_settings(settings, schema):
    saved = schema.get("jump_task")
    if saved is None:
        # Historical v2 jump checkpoints used the original timer-based helper.
        settings.jump.implementation = "legacy"
        settings.rewards.weights.update(jump_push_velocity=4., jump_push_height=2.,
            jump_success=100., jump_takeoff=0., jump_failed=0.)
        return
    if saved.get("version") != "measured-jump-v1":
        raise ValueError("Unsupported jump checkpoint version")
    for group in ("commands", "jump"):
        target = getattr(settings, group)
        for name, value in saved[group].items():
            if not hasattr(target, name):
                raise ValueError(f"Unsupported saved jump setting: {group}.{name}")
            setattr(target, name, tuple(value) if isinstance(getattr(target, name), tuple) else value)
