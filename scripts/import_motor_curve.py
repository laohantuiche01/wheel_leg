"""Convert the old measured motor-shaft CSV into a standalone output-side JSON."""
import argparse
import json
from pathlib import Path
import bootstrap
from wheel_legged_lab.components.legacy_control import TorqueSpeedCurve

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input_csv", type=Path, required=True)
parser.add_argument("--output_json", type=Path, required=True)
parser.add_argument("--group", required=True, help="Motor group name from the new robot JSON, e.g. wheels")
parser.add_argument("--gear_ratio", type=float, required=True)
parser.add_argument("--efficiency", type=float, default=1.0)
args = parser.parse_args()
points = TorqueSpeedCurve.read_csv(args.input_csv, gear_ratio=args.gear_ratio, efficiency=args.efficiency)
TorqueSpeedCurve(points, "cpu")  # validate before creating a file
with args.output_json.open("x") as stream:
    json.dump({args.group: points}, stream, indent=2)
    stream.write("\n")
print(f"Converted {len(points)} output-side points into {args.output_json}")
