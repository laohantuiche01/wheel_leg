"""Inspect the generated USD and configuration without launching simulation."""
import hashlib
import json
from pathlib import Path
import re

from usd_python import enable_usd

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src/wheel_legged_lab/assets"


def main():
    enable_usd()
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    spec = json.loads((ASSETS / "v2_0.json").read_text())
    stage = Usd.Stage.Open(str(ASSETS / spec["usd"]))
    cache = UsdGeom.XformCache()
    bodies = {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    joints = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)}
    movable = {name: p for name, p in joints.items() if (p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint))
               and not UsdPhysics.Joint(p).GetExcludeFromArticulationAttr().Get()}
    assert set(movable) == set(spec["joint_names"]) == set(spec["rest_joint_positions"])
    assert all(value == 0 for value in spec["rest_joint_positions"].values())
    owners = [name for group in spec["motors"] for name in group["joints"]] + spec["spring_joints"] + spec["auxiliary_joints"]
    for pattern in spec["passive_joint_patterns"]:
        matches = [name for name in movable if re.fullmatch(pattern, name)]
        assert matches, pattern
        owners.extend(matches)
    assert len(owners) == len(set(owners)) == len(movable) and set(owners) == set(movable)
    assert len(bodies) == 32 and len(movable) == 30
    assert all(name in bodies for name in spec["wheel_bodies"] + spec["termination_bodies"])
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert roots == [bodies[spec["base_body"]]]
    assert cache.GetLocalToWorldTransform(roots[0]).ExtractTranslation().GetLength() < 1e-9
    assert all(p.IsA(UsdPhysics.PrismaticJoint)
               for name, p in movable.items() if name in spec["spring_joints"])

    max_anchor_error = 0.0
    closure_count = 0
    for name, prim in joints.items():
        joint = UsdPhysics.Joint(prim)
        closure_count += bool(joint.GetExcludeFromArticulationAttr().Get())
        assert not any(p.GetName().startswith("drive:") for p in prim.GetProperties()), name
        frames = []
        for i in (0, 1):
            body_path = prim.GetRelationship(f"physics:body{i}").GetTargets()[0]
            transform = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body_path))
            pos = transform.Transform(Gf.Vec3d(prim.GetAttribute(f"physics:localPos{i}").Get()))
            rot = Gf.Rotation(Gf.Quatd(prim.GetAttribute(f"physics:localRot{i}").Get())) * transform.ExtractRotation()
            frames.append((pos, rot))
        error = (frames[0][0] - frames[1][0]).GetLength()
        angle = (frames[0][1].GetInverse() * frames[1][1]).GetAngle()
        assert error < 1e-6 and min(angle, abs(360 - angle)) < 1e-3, name
        max_anchor_error = max(max_anchor_error, error)
        if name in movable:
            lower, upper = (prim.GetAttribute(f"physics:{limit}Limit").Get() for limit in ("lower", "upper"))
            assert lower <= 0 <= upper, name
    assert closure_count == 6

    clearances = []
    for name in spec["wheel_bodies"]:
        wheel = bodies[name]
        center_z = cache.GetLocalToWorldTransform(wheel).ExtractTranslation()[2]
        clearance = spec["rest_height"] + center_z - spec["wheel_radius"]
        assert abs(clearance) < 1e-6, (name, clearance)
        clearances.append(clearance)

    # Independently verify the direction map against both assets' joint frames.
    legacy = Usd.Stage.Open(str(ASSETS / "wheelbipe_v1_0.usd"))
    old_spec = json.loads((ASSETS / "v1_0.json").read_text())
    old_joints = {p.GetName(): p for p in legacy.Traverse() if p.IsA(UsdPhysics.Joint)}
    def axis(prim, owner):
        j = UsdPhysics.Joint(prim)
        parent = owner.GetPrimAtPath(j.GetBody0Rel().GetTargets()[0])
        frame = Gf.Rotation(Gf.Quatd(j.GetLocalRot0Attr().Get())) * cache.GetLocalToWorldTransform(parent).ExtractRotation()
        index = {"X": 0, "Y": 1, "Z": 2}[prim.GetAttribute("physics:axis").Get()]
        vector = Gf.Vec3d(0)
        vector[index] = 1
        return frame.TransformDir(vector)
    for old_group, group in zip(old_spec["motors"], spec["motors"], strict=True):
        for old_name, name, direction in zip(old_group["joints"], group["joints"], group["directions"], strict=True):
            assert (axis(old_joints[old_name], legacy) - direction * axis(joints[name], stage)).GetLength() < 1e-5, name

    report = dict(status="pass", rigid_bodies=len(bodies), dof_count=len(movable), loop_constraints=closure_count,
        max_anchor_error_m=max_anchor_error, wheel_clearances_m=clearances,
        action_directions_verified=True, joint_ownership_verified=True, imported_drives_removed=True,
        asset_sha256=hashlib.sha256((ASSETS / spec["usd"]).read_bytes()).hexdigest(),
        simulator_started=False, training_started=False)
    (ROOT / "MODEL_CHECK.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
