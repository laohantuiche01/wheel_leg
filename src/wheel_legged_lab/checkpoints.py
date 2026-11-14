"""Checkpoint contract validation before weights are loaded."""
import json
from pathlib import Path
import warnings


def transfer_locomotion_actor(policy, path, schema, *, device="cpu"):
    """Seed walk/spin/jump from a compatible v2 actor; keep critic/optimizer fresh.

    Spin appends five fields to EACH history frame. Copy the old frame blocks
    into their matching positions, zeroing the new columns so the initial mean
    policy is preserved. This is explicit transfer, never an exact resume.
    """
    from dataclasses import asdict
    import hashlib
    import torch
    from .settings import MotorSettings

    path = Path(path)
    actual = json.loads((path.parent / "observation_schema.json").read_text())
    expected = json.loads(json.dumps(schema))
    if expected.get("motion_task", {}).get("task") not in ("walk", "spin") and not expected.get("jump_task"):
        raise ValueError("Actor transfer is only available for walk/spin/measured-jump destinations")
    required = ("version", "robot", "asset_sha256", "action_names", "action_directions", "motor_modes",
                "action_scales", "motor_groups", "rest_joint_positions", "rest_height", "wheel_radius", "spring_settings")
    for key in required:
        if key not in actual or actual[key] != expected.get(key):
            raise ValueError(f"Actor transfer requires matching {key}")
    defaults = json.loads(json.dumps(asdict(MotorSettings())))
    if actual.get("motor_settings", defaults) != expected.get("motor_settings", defaults):
        raise ValueError("Actor transfer requires matching motor settings; choose the source strategy")
    if actual.get("auxiliary_holds", {}) != expected.get("auxiliary_holds", {}):
        raise ValueError("Actor transfer requires matching auxiliary pitch control")
    for value in (actual, expected):
        value.setdefault("observation_settings", {}).pop("noise_enabled", None)
        value["observation_settings"].setdefault("actor_history", 1)
        value["observation_settings"].setdefault("actor_delay", 0)
    if actual["observation_settings"] != expected["observation_settings"]:
        raise ValueError("Actor transfer requires matching base observation scales, delay and history")
    if actual.get("actor_blocks") != expected.get("actor_blocks"):
        raise ValueError("Base actor block semantics differ")
    history = expected["observation_settings"]["actor_history"]
    base_width = 17 + 3 * len(expected["action_names"])
    source_extra = 5 if actual.get("gimbal_observation") else 0
    old_width = base_width + source_extra
    if actual["actor_dim"] != old_width * history:
        raise ValueError("Unsupported source actor layout")
    new_width = expected["actor_dim"] // history
    if new_width not in (base_width, base_width + 5) or expected["actor_dim"] != new_width * history:
        raise ValueError("Unsupported destination actor layout")
    if source_extra:
        if (new_width != old_width or actual.get("gimbal_observation") != expected.get("gimbal_observation")
                or actual.get("motion_task", {}).get("task") != "spin"
                or expected.get("motion_task", {}).get("task") != "spin"
                or not actual.get("motion_task", {}).get("spin")
                or actual["motion_task"]["spin"] != expected["motion_task"].get("spin")):
            raise ValueError("Augmented actor transfer requires matching spin observations and gimbal control")

    source = torch.load(path, map_location=device, weights_only=False)["model_state_dict"]
    target = policy.state_dict()
    actor_keys = [key for key in target if key.startswith("actor.")]
    if not actor_keys or set(actor_keys) != {key for key in source if key.startswith("actor.")}:
        raise ValueError("Actor layer names differ")
    if any(key.startswith("actor_obs_normalizer.") for key in source):
        raise ValueError("Normalized actor transfer needs a separate normalization mapping")
    for key in actor_keys:
        old = source[key]
        if key == "actor.0.weight" and old_width != new_width:
            if old.shape != (target[key].shape[0], old_width * history):
                raise ValueError("Actor first-layer width differs")
            expanded = torch.zeros_like(target[key])
            for frame in range(history):
                expanded[:, frame * new_width:frame * new_width + old_width] = old[:, frame * old_width:(frame + 1) * old_width]
            target[key] = expanded
        elif old.shape == target[key].shape:
            target[key] = old
        else:
            raise ValueError(f"Actor hidden layer shape differs: {key}")
    phase_columns = []
    if (expected.get("jump_task", {}).get("version") == "measured-jump-v2"
            and actual.get("jump_task", {}).get("version") != "measured-jump-v2"):
        # These two slots were always zero in the older seven-field task mode.
        # Start their new phase signals neutrally rather than activating random
        # first-layer coefficients that were never trained on a nonzero input.
        phase_columns = [frame * new_width + base_width - 6 + offset
                         for frame in range(history) for offset in (0, 1)]
        target["actor.0.weight"] = target["actor.0.weight"].clone()
        target["actor.0.weight"][:, phase_columns] = 0
    policy.load_state_dict(target, strict=True)
    return dict(source=str(path.resolve()), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_actor_dim=actual["actor_dim"], destination_actor_dim=expected["actor_dim"], history=history,
        zeroed_new_columns=(new_width - old_width) * history, zeroed_phase_columns=phase_columns,
        critic="fresh initialization", optimizer="fresh initialization")

def save_schema(directory, schema):
    path = Path(directory) / "observation_schema.json"
    path.write_text(json.dumps(schema, indent=2) + "\n")

def validate_checkpoint(path, schema, *, allow_auxiliary_change=False, allow_control_change=False):
    path = Path(path)
    manifest = path.parent / "observation_schema.json"
    if not manifest.is_file():
        raise ValueError("Checkpoint has no observation schema. Legacy weights require explicit migration.")
    expected = json.loads(json.dumps(schema))
    actual = json.loads(manifest.read_text())
    # Noise can be disabled during deterministic evaluation without changing layout.
    for value in (expected, actual):
        value.get("observation_settings", {}).pop("noise_enabled", None)
        if "observation_settings" in value:
            value["observation_settings"].setdefault("actor_history", 1)
            value["observation_settings"].setdefault("actor_delay", 0)
    # Old v2 manifests used passive damping for every auxiliary joint. This is
    # the only controlled compatibility exception: asset/actions/observations
    # must still match exactly before old weights can seed the new controller.
    expected_holds = expected.pop("auxiliary_holds", {})
    actual_holds = actual.pop("auxiliary_holds", {})
    if allow_control_change:
        expected.pop("motor_settings", None)
        actual.pop("motor_settings", None)
        for value in (expected, actual):
            value.get("observation_settings", {}).pop("actor_delay", None)
    if actual != expected:
        raise ValueError("Checkpoint robot/action/observation schema does not match this environment")
    if actual_holds != expected_holds:
        if not allow_auxiliary_change:
            raise ValueError("Checkpoint auxiliary control differs. For explicit evaluation or fine-tuning only, "
                             "use --allow_auxiliary_change; asset/action/observation checks still apply.")
        warnings.warn("Loading weights across an auxiliary-control change; this is not an exact training resume.",
                      UserWarning, stacklevel=2)
    if allow_control_change:
        warnings.warn("Explicit control-change fine-tuning: motor processing/observation delay may differ; "
                      "asset, joint gains, action layout and history dimensions must still match.",
                      UserWarning, stacklevel=2)
