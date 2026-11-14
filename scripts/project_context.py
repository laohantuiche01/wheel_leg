"""Read-only onboarding snapshot. Standard library only; never imports the project/Kit."""
import argparse
import ast
import hashlib
import json
import operator
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def default_robot_spec(root):
    """Resolve the declared default without executing configuration code."""
    source = root / 'src/wheel_legged_lab/settings.py'
    tree = ast.parse(source.read_text())
    def resolve(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return Path(node.value)
        if isinstance(node, ast.Name) and node.id == 'ASSET_DIR':
            return source.parent / 'assets'
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return resolve(node.left) / resolve(node.right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('str', 'Path') and len(node.args) == 1:
            return resolve(node.args[0])
        raise ValueError('Default robot expression is dynamic; provide --robot-spec explicitly')
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'TaskSettings':
            for field in node.body:
                if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name) and field.target.id == 'robot_file':
                    path = resolve(field.value)
                    return (root / path).resolve() if not path.is_absolute() else path.resolve()
    raise ValueError('TaskSettings.robot_file not found; provide --robot-spec explicitly')


def dimensions(root, action_count, history):
    """Evaluate only arithmetic in the actual schema expressions, never eval()."""
    tree = ast.parse((root / 'src/wheel_legged_lab/components/observations.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'observation_schema')
    schema = next(n for n in ast.walk(fn) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name) and n.func.id == 'dict')
    expressions = {n.arg: n.value for n in schema.keywords}
    extra = next(n.value for n in fn.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'extra' for t in n.targets))
    spin_extra = ast.literal_eval(extra.body)
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}
    def arithmetic(node, values):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](arithmetic(node.left, values), arithmetic(node.right, values))
        raise ValueError('Unsupported observation expression; inspect observations.py directly')
    result = {}
    for name, amount in (('base', 0), ('spin', spin_extra)):
        values = dict(a=action_count, history=history, extra=amount)
        result[name] = {key: arithmetic(expressions[key], values) for key in ('actor_dim', 'critic_dim')}
    return result


def timing(root):
    tree = ast.parse((root / 'src/wheel_legged_lab/config.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WheelLeggedEnvCfg')
    decimation = next(ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'decimation' for t in n.targets))
    call = next(n for n in ast.walk(cls) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == 'SimulationCfg')
    dt = next(ast.literal_eval(k.value) for k in call.keywords if k.arg == 'dt')
    return dict(physics_dt=dt, decimation=decimation, control_dt=dt * decimation)


def evidence_state(root, registry, robot_name, asset_hash, spec_hash):
    changed = []
    for relative, expected in registry.get('source_fingerprints', {}).items():
        path = root / relative
        if not path.is_file() or digest(path) != expected:
            changed.append(relative)
    entries = []
    for row in registry.get('evidence', []):
        checkpoint = root / row['checkpoint']
        report = root / row['report']
        applies = (row['robot'] == robot_name and row.get('asset_sha256') == asset_hash
                   and row.get('spec_sha256') == spec_hash)
        entries.append(dict(id=row['id'], task=row['task'], checkpoint=row['checkpoint'],
            checkpoint_exists=checkpoint.is_file(), report_exists=report.is_file(),
            checkpoint_hash_matches=(digest(checkpoint) == row['checkpoint_sha256']) if checkpoint.is_file() else False,
            applies_to_selected_robot_contract=applies, result=row['result'], limits=row['limits']))
    return dict(snapshot_present=bool(registry), recorded_on=registry.get('recorded_on'), changed_since_handoff=changed, entries=entries)


def checkpoint_precheck(root, checkpoint, spec, asset_hash, dims):
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f'Checkpoint not found: {checkpoint}')
    schema = read_json(checkpoint.parent / 'observation_schema.json')
    groups = spec['motors']
    expected = dict(robot=spec['name'], asset_sha256=asset_hash,
        action_names=[j for g in groups for j in g['joints']],
        action_directions=[d for g in groups for d in g.get('directions', [1.] * len(g['joints']))],
        motor_modes=[g['mode'] for g in groups], action_scales=[g['action_scale'] for g in groups],
        motor_groups=[{**g, 'directions': g.get('directions', [1.] * len(g['joints']))} for g in groups],
        auxiliary_holds=spec.get('auxiliary_holds', {}),
        rest_joint_positions=spec['rest_joint_positions'], rest_height=spec['rest_height'], wheel_radius=spec['wheel_radius'])
    launch_path = checkpoint.parent / 'launch.json'
    launch = read_json(launch_path) if launch_path.is_file() else {}
    task = 'jump' if schema.get('jump_task') else schema.get('motion_task', {}).get('task', launch.get('task'))
    family = 'spin' if schema.get('gimbal_observation') else 'base'
    expected.update(dims[family])
    differences = [name for name, value in expected.items() if schema.get(name) != value]
    return dict(path=str(checkpoint), task=task, strategy=launch.get('strategy', 'unknown'),
        jump_version=schema.get('jump_task', {}).get('version'),
        history=schema.get('observation_settings', {}).get('actor_history', 1),
        status='mismatch' if differences else 'partial_precheck_pass', mismatched_fields=differences,
        authority='Partial triage only. Full motor/spring/observation/task compatibility is enforced by checkpoints.validate_checkpoint or explicit actor transfer.')


def build_context(root=ROOT, robot_spec=None, checkpoint=None, include_recent=False):
    root = Path(root).expanduser().resolve()
    spec_path = Path(robot_spec).expanduser().resolve() if robot_spec else default_robot_spec(root)
    spec = read_json(spec_path)
    asset = (spec_path.parent / spec['usd']).resolve()
    if not asset.is_file():
        raise ValueError(f'Robot asset not found: {asset}')
    asset_hash = digest(asset)
    history = 1
    if checkpoint:
        saved = read_json(Path(checkpoint).expanduser().resolve().parent / 'observation_schema.json')
        history = saved.get('observation_settings', {}).get('actor_history', 1)
    groups = spec['motors']
    actions = [j for g in groups for j in g['joints']]
    warnings = []
    try:
        dims = dimensions(root, len(actions), history)
    except (ValueError, StopIteration, KeyError, TypeError) as exc:
        dims = None
        warnings.append(str(exc))
    try:
        steps = timing(root)
    except (ValueError, StopIteration, KeyError, TypeError) as exc:
        steps = None
        warnings.append(f'Could not resolve timing: {exc}')
    registry_path = root / 'docs/PROJECT_STATE.json'
    registry = read_json(registry_path) if registry_path.is_file() else {}
    migration_path = root / 'MODEL_MIGRATION.json'
    migration = read_json(migration_path) if migration_path.is_file() else {}
    result = dict(repo=str(root), selected_spec=str(spec_path), spec_sha256=digest(spec_path), robot=spec['name'], asset=str(asset), asset_sha256=asset_hash,
        action_count=len(actions), action_names=actions,
        action_directions=[d for g in groups for d in g.get('directions', [1.] * len(g['joints']))],
        motor_groups=[{k: g.get(k) for k in ('name', 'mode', 'action_scale', 'kp', 'kd', 'effort_limit', 'speed_limit')} for g in groups],
        declared_joint_count=len(spec.get('joint_names', [])), wheel_bodies=spec['wheel_bodies'],
        rest_height=spec['rest_height'], wheel_radius=spec['wheel_radius'], dimensions=dims, history=history, timing=steps,
        evidence=evidence_state(root, registry, spec['name'], asset_hash, digest(spec_path)), warnings=warnings,
        asset_check_scope=dict(migration_spec=migration.get('spec'),
            selected_is_v2_check_target=spec_path == (root / 'src/wheel_legged_lab/assets/v2_0.json').resolve(),
            note='check_model.py and migrate_v2_model.py currently contain v2-specific names/counts/zero-pose assumptions; they are not generic new-robot validation.'),
        read_next=['AGENTS.md', 'docs/PROJECT_CONTEXT.md', 'skills/iassclab-maintainer/SKILL.md'],
        simulator_started=False, files_written=False)
    if checkpoint:
        if dims is None:
            result['checkpoint'] = dict(status='unknown', note='Resolve observation layout before a compatibility decision')
        else:
            result['checkpoint'] = checkpoint_precheck(root, checkpoint, spec, asset_hash, dims)
    if include_recent:
        runs = []
        for path in (root / 'logs').rglob('observation_schema.json'):
            files = [p for p in path.parent.glob('model_*.pt') if p.stem.removeprefix('model_').isdigit()]
            if files:
                latest = max(files, key=lambda p: int(p.stem.removeprefix('model_')))
                runs.append((path.parent.stat().st_mtime, str(latest.relative_to(root))))
        result['recent_checkpoints_inventory_only'] = [p for _, p in sorted(runs, reverse=True)[:6]]
    try:
        result['git_head'] = subprocess.run(['git', '--no-optional-locks', '-C', str(root), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        result['git_status'] = subprocess.run(['git', '--no-optional-locks', '-C', str(root), 'status', '--short'], capture_output=True, text=True, check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        result['git_head'] = 'unavailable'
    return result


def format_summary(data):
    lines = [f"Repo: {data['repo']}", f"Robot: {data['robot']}", f"Spec: {data['selected_spec']}",
        f"Asset SHA256: {data['asset_sha256']}", f"Actions ({data['action_count']}): {', '.join(data['action_names'])}",
        f"Directions: {data['action_directions']}", f"Dimensions: {data['dimensions']}; history={data['history']}",
        f"Timing: {data['timing']}"]
    for group in data['motor_groups']:
        lines.append(f"Motor group: {group}")
    if data.get('checkpoint'):
        lines.append('Checkpoint triage: ' + json.dumps(data['checkpoint'], ensure_ascii=False))
    changed = data['evidence']['changed_since_handoff']
    lines.append(f"Knowledge snapshot present: {data['evidence']['snapshot_present']}; behavior files changed: {changed}")
    for row in data['evidence']['entries']:
        lines.append(f"Evidence {row['id']}: contract_applies={row['applies_to_selected_robot_contract']}, "
            f"checkpoint_matches={row['checkpoint_hash_matches']}, report_exists={row['report_exists']}; {row['result']}; limits={row['limits']}")
    lines += ['Asset checker scope: ' + data['asset_check_scope']['note'],
              'Next read: ' + ', '.join(data['read_next']),
              'This command is read-only, starts no simulator and does not certify physics or full checkpoint compatibility.']
    if data['warnings']:
        lines.append('Warnings: ' + '; '.join(data['warnings']))
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--robot-spec', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--recent', action='store_true', help='List recent log checkpoints as inventory, not recommendations')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    try:
        data = build_context(args.repo, args.robot_spec, args.checkpoint, args.recent)
    except (ValueError, OSError, KeyError, SyntaxError) as exc:
        parser.error(str(exc))
    print(json.dumps(data, ensure_ascii=False, indent=2) if args.json else format_summary(data))
    return 3 if data.get('checkpoint', {}).get('status') == 'mismatch' else 0


if __name__ == '__main__':
    raise SystemExit(main())
