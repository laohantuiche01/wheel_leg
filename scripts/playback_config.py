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
    args.task = args.task or launch.get("task", "stand")
    if args.task not in ("stand", "height", "velocity", "jump"):
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
    settings.observations.noise_enabled = False
    settings.validate()
