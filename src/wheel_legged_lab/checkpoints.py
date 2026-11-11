"""Checkpoint contract validation before weights are loaded."""
import json
from pathlib import Path

def save_schema(directory, schema):
    path = Path(directory) / "observation_schema.json"
    path.write_text(json.dumps(schema, indent=2) + "\n")

def validate_checkpoint(path, schema):
    path = Path(path)
    manifest = path.parent / "observation_schema.json"
    if not manifest.is_file():
        raise ValueError("Checkpoint has no observation schema. Legacy weights require explicit migration.")
    expected = json.loads(json.dumps(schema))
    actual = json.loads(manifest.read_text())
    # Noise can be disabled during deterministic evaluation without changing layout.
    for value in (expected, actual):
        value.get("observation_settings", {}).pop("noise_enabled", None)
    if actual != expected:
        raise ValueError("Checkpoint robot/action/observation schema does not match this environment")
