"""Static-only source/manifest validation. Never imports project or simulator modules."""
import ast
import hashlib
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/wheel_legged_lab"
errors = []
files = sorted(ROOT.rglob("*.py"))
trees = {}
for path in files:
    try:
        trees[path] = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as exc:
        errors.append(str(exc))
for path, tree in trees.items():
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
            if node.level and path.is_relative_to(PACKAGE):
                parent = path.parent
                for _ in range(node.level - 1):
                    parent = parent.parent
                target = parent.joinpath(*(node.module or "").split("."))
                if not (target.with_suffix(".py").is_file() or (target / "__init__.py").is_file()):
                    errors.append(f"Missing relative import in {path}: {node.module}")
        else:
            continue
        for module in modules:
            if module.startswith(("agent_tasks", "agent_world", "agent_rl")) or any(part in module for part in ("wheelbipe_V13", "wheelbipe_V14", "wheelbipe25_v3")):
                errors.append(f"Legacy dependency: {path}: {module}")
            if module.startswith("wheel_legged_lab."):
                target = ROOT / "src" / module.replace(".", "/")
                if not (target.with_suffix(".py").is_file() or (target / "__init__.py").is_file()):
                    errors.append(f"Missing package import: {module}")
env_tree = trees[PACKAGE / "env.py"]
classes = [n for n in env_tree.body if isinstance(n, ast.ClassDef)]
if len(classes) != 1 or [ast.unparse(b) for b in classes[0].bases] != ["DirectRLEnv"]:
    errors.append("Environment must be a single direct DirectRLEnv subclass")
settings_tree = trees[PACKAGE / "settings.py"]
reward_cfg = next(n for n in settings_tree.body if isinstance(n, ast.ClassDef) and n.name == "RewardSettings")
weights = next(n for n in ast.walk(reward_cfg) if isinstance(n, ast.Lambda)).body
weight_names = {ast.literal_eval(key) for key in weights.keys}
reward_tree = trees[PACKAGE / "components/rewards.py"]
term_dict = next(n.value for n in ast.walk(reward_tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "terms" for t in n.targets))
term_names = {ast.literal_eval(key) for key in term_dict.keys}
if weight_names - term_names:
    errors.append(f"Reward keys not implemented: {weight_names - term_names}")
manifest = json.loads((ROOT / "ASSET_PROVENANCE.json").read_text())
legacy = json.loads((PACKAGE / "assets/v1_0.json").read_text())
legacy_hash = hashlib.sha256((PACKAGE / "assets" / legacy["usd"]).read_bytes()).hexdigest()
if legacy_hash != manifest["sha256"]:
    errors.append("USD copy hash does not match the source manifest")
migration = json.loads((ROOT / "MODEL_MIGRATION.json").read_text())
model = json.loads((ROOT / migration["spec"]).read_text())
asset_hash = hashlib.sha256((ROOT / migration["asset"]).read_bytes()).hexdigest()
for name in ("source", "asset", "spec"):
    if hashlib.sha256((ROOT / migration[name]).read_bytes()).hexdigest() != migration[name + "_sha256"]:
        errors.append(f"v2 {name} checksum mismatch; rerun migration/validation")
for name, key in (("source", "v2_source_sha256"), ("asset", "v2_converted_sha256"), ("spec", "v2_robot_spec_sha256")):
    if manifest.get(key) != migration[name + "_sha256"]:
        errors.append(f"Asset provenance disagrees with v2 migration: {key}")
for name, checksum in migration["preserved_code_sha256"].items():
    if hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() != checksum:
        errors.append(f"Preserved scheduling/reward/PPO code changed: {name}")
if len(model["joint_names"]) != migration["dof_count"] or set(model["joint_names"]) != set(model["rest_joint_positions"]):
    errors.append("v2 rest pose must explicitly cover all articulation DOFs")
with (ROOT / "pyproject.toml").open("rb") as stream:
    project = tomllib.load(stream)
report = dict(status="pass" if not errors else "fail", checks="AST, import paths, inheritance, reward keys, JSON/TOML, v1/v2 checksums, preserved scheduling/PPO, v2 rest coverage",
              python_files=len(files), python_lines=sum(len(path.read_text().splitlines()) for path in files),
              environment_class="WheelLeggedEnv(DirectRLEnv)", action_count=sum(len(g["joints"]) for g in model["motors"]),
              asset_sha256=asset_hash, project_version=project["project"]["version"],
              robot=model["name"], dof_count=migration["dof_count"], loop_constraint_count=migration["loop_constraint_count"],
              simulator_started=False, smoke_test_run=False, training_started=False, errors=errors)
(ROOT / "STATIC_CHECK.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(bool(errors))
