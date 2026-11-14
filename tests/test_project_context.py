"""Onboarding remains read-only and detects same-shape robot/metadata changes."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('project_context',ROOT/'scripts/project_context.py')
context=importlib.util.module_from_spec(spec);spec.loader.exec_module(context)


class ProjectContextTest(unittest.TestCase):
    def fixture(self,root):
        package=root/'src/wheel_legged_lab';assets=package/'assets';assets.mkdir(parents=True)
        (package/'__init__.py').write_text('raise RuntimeError("onboarding must never import the project")\n')
        (package/'settings.py').write_text("from pathlib import Path\nASSET_DIR=Path(__file__).parent/'assets'\nclass TaskSettings:\n    robot_file: str=str(ASSET_DIR/'v2_0.json')\n")
        (package/'components').mkdir()
        for relative in ('config.py','components/observations.py'):
            shutil.copyfile(ROOT/'src/wheel_legged_lab'/relative,package/relative)
        robot=json.loads((ROOT/'src/wheel_legged_lab/assets/v2_0.json').read_text())
        robot['usd']='robot.usd';asset=assets/'robot.usd';asset.write_bytes(b'fixture asset, never open USD')
        path=assets/'v2_0.json';path.write_text(json.dumps(robot))
        run=root/'logs/test';run.mkdir(parents=True)
        checkpoint=run/'model_1.pt';checkpoint.write_bytes(b'not a pickle; context never deserializes weights')
        schema=dict(robot=robot['name'],asset_sha256=context.digest(asset),
            action_names=[j for g in robot['motors'] for j in g['joints']],
            action_directions=[d for g in robot['motors'] for d in g['directions']],
            motor_modes=[g['mode'] for g in robot['motors']],action_scales=[g['action_scale'] for g in robot['motors']],
            motor_groups=robot['motors'],auxiliary_holds=robot['auxiliary_holds'],
            rest_joint_positions=robot['rest_joint_positions'],rest_height=robot['rest_height'],wheel_radius=robot['wheel_radius'],
            actor_dim=70,critic_dim=60,observation_settings=dict(actor_history=2))
        (run/'observation_schema.json').write_text(json.dumps(schema))
        (run/'launch.json').write_text(json.dumps(dict(task='stand',strategy='legacy_ppo')))
        docs=root/'docs';docs.mkdir();(docs/'report.json').write_text('{}')
        registry=dict(recorded_on='test',source_fingerprints={'src/wheel_legged_lab/settings.py':context.digest(package/'settings.py')},
            evidence=[dict(id='seed',robot=robot['name'],asset_sha256=context.digest(asset),spec_sha256=context.digest(path),task='stand',
                checkpoint='logs/test/model_1.pt',checkpoint_sha256=context.digest(checkpoint),report='docs/report.json',result='fixture',limits='test only')])
        (docs/'PROJECT_STATE.json').write_text(json.dumps(registry))
        return path,checkpoint,robot

    def test_reads_actual_defaults_and_history_without_importing_or_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,checkpoint,_=self.fixture(root)
            before={str(p.relative_to(root)):context.digest(p) for p in root.rglob('*') if p.is_file()}
            data=context.build_context(root,checkpoint=checkpoint)
            self.assertEqual(data['selected_spec'],str(path))
            self.assertEqual(data['dimensions']['base'],dict(actor_dim=70,critic_dim=60))
            self.assertEqual(data['timing']['control_dt'],.02)
            self.assertEqual(data['checkpoint']['status'],'partial_precheck_pass')
            self.assertTrue(data['evidence']['entries'][0]['applies_to_selected_robot_contract'])
            after={str(p.relative_to(root)):context.digest(p) for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before,after)

    def test_new_same_dimension_robot_invalidates_checkpoint_and_prior_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,checkpoint,robot=self.fixture(root)
            changed=deepcopy(robot);old=changed['motors'][0]['joints'][0];new='new_left_joint'
            changed['motors'][0]['joints'][0]=new;changed['motors'][0]['directions'][0]*=-1
            changed['motors'][0]['kp']=90.
            changed['joint_names']=[new if x==old else x for x in changed['joint_names']]
            changed['rest_joint_positions'][new]=changed['rest_joint_positions'].pop(old)
            replacement=path.with_name('new_robot.json');replacement.write_text(json.dumps(changed))
            settings=root/'src/wheel_legged_lab/settings.py'
            settings.write_text(settings.read_text().replace("'v2_0.json'","'new_robot.json'"))
            data=context.build_context(root,checkpoint=checkpoint)
            self.assertEqual(data['selected_spec'],str(replacement))
            self.assertEqual(data['action_count'],6)
            self.assertEqual(data['checkpoint']['status'],'mismatch')
            self.assertIn('action_names',data['checkpoint']['mismatched_fields'])
            self.assertIn('action_directions',data['checkpoint']['mismatched_fields'])
            self.assertIn('motor_groups',data['checkpoint']['mismatched_fields'])
            self.assertFalse(data['evidence']['entries'][0]['applies_to_selected_robot_contract'])
            self.assertIn('src/wheel_legged_lab/settings.py',data['evidence']['changed_since_handoff'])

    def test_same_asset_pd_edit_and_missing_or_modified_evidence_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,checkpoint,robot=self.fixture(root)
            robot['motors'][0]['kd']=3.;path.write_text(json.dumps(robot))
            checkpoint.write_bytes(b'changed weights');(root/'docs/report.json').unlink()
            data=context.build_context(root,checkpoint=checkpoint)
            row=data['evidence']['entries'][0]
            self.assertFalse(row['applies_to_selected_robot_contract'])
            self.assertFalse(row['checkpoint_hash_matches']);self.assertFalse(row['report_exists'])
            self.assertIn('motor_groups',data['checkpoint']['mismatched_fields'])

    def test_dynamic_default_needs_explicit_robot_not_a_v2_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,_,_=self.fixture(root)
            (root/'src/wheel_legged_lab/settings.py').write_text('class TaskSettings:\n    robot_file: str=choose_robot()\n')
            with self.assertRaisesRegex(ValueError,'--robot-spec'):
                context.build_context(root)
            self.assertEqual(context.build_context(root,robot_spec=path)['selected_spec'],str(path))

if __name__=='__main__':unittest.main()
