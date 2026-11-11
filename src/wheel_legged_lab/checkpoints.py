"""Checkpoint contract validation before weights are loaded."""
import json
from pathlib import Path
import warnings

def save_schema(directory, schema):
    path = Path(directory) / "observation_schema.json"
    path.write_text(json.dumps(schema, indent=2) + "\n")

def validate_checkpoint(path, schema, *, allow_auxiliary_change=False):
    path = Path(path)
    manifest = path.parent / "observation_schema.json"
    if not manifest.is_file():
        raise ValueError("Checkpoint has no observation schema. Legacy weights require explicit migration.")
    expected = json.loads(json.dumps(schema))
    actual = json.loads(manifest.read_text())
    # Noise can be disabled during deterministic evaluation without changing layout.
    for value in (expected, actual):
        value.get("observation_settings", {}).pop("noise_enabled", None)
    # Old v2 manifests used passive damping for every auxiliary joint. This is
    # the only controlled compatibility exception: asset/actions/observations
    # must still match exactly before old weights can seed the new controller.
    expected_holds = expected.pop("auxiliary_holds", {})
    actual_holds = actual.pop("auxiliary_holds", {})
    if actual != expected:
        raise ValueError("Checkpoint robot/action/observation schema does not match this environment")
    if actual_holds != expected_holds:
        if not allow_auxiliary_change:
            raise ValueError("Checkpoint auxiliary control differs. For explicit evaluation or fine-tuning only, "
                             "use --allow_auxiliary_change; asset/action/observation checks still apply.")
        warnings.warn("Loading weights across an auxiliary-control change; this is not an exact training resume.",
                      UserWarning, stacklevel=2)
