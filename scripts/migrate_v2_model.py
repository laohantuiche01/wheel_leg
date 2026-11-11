"""Build a standalone v2 asset and explicit joint contract; no simulator is started.

Run with the existing Isaac Python environment. The source USDZ is never modified.
This is an asset/config migration, not a trained-network weight conversion.
"""
import hashlib
import json
from pathlib import Path

from usd_python import enable_usd

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src/wheel_legged_lab/assets"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    enable_usd()
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    source = ROOT / "model/v2_0.usdz"
    stage = Usd.Stage.Open(str(source))
    root = stage.GetDefaultPrim()
    if str(root.GetPath()) != "/generated_robot":
        raise ValueError("Unexpected v2 default prim")
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or UsdGeom.GetStageUpAxis(stage) != "Z":
        raise ValueError("Expected meter units and Z up")
    cache = UsdGeom.XformCache()
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    movable = [p for p in joints if (p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint))
               and not UsdPhysics.Joint(p).GetExcludeFromArticulationAttr().Get()]
    closures = [p for p in joints if UsdPhysics.Joint(p).GetExcludeFromArticulationAttr().Get()]
    base = stage.GetPrimAtPath(root.GetPath().AppendChild("base_link"))
    base_transform = cache.GetLocalToWorldTransform(base)
    if base_transform.ExtractRotation().GetAngle() > 1e-5:
        raise ValueError("Expected an unrotated base frame")
    base_translation = base_transform.ExtractTranslation()

    # The CAD pose is the v2 zero pose. Check every anchor and joint frame before
    # using zeros, including the six loop closures excluded from the articulation.
    joint_report = []
    for prim in joints:
        joint = UsdPhysics.Joint(prim)
        frames = []
        for i in (0, 1):
            target = prim.GetRelationship(f"physics:body{i}").GetTargets()
            if len(target) != 1:
                raise ValueError(f"Missing joint endpoint: {prim.GetPath()}")
            body_transform = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(target[0]))
            pos = body_transform.Transform(Gf.Vec3d(prim.GetAttribute(f"physics:localPos{i}").Get()))
            rot = Gf.Rotation(Gf.Quatd(prim.GetAttribute(f"physics:localRot{i}").Get())) * body_transform.ExtractRotation()
            frames.append((pos, rot))
        error = (frames[1][0] - frames[0][0]).GetLength()
        angle = (frames[0][1].GetInverse() * frames[1][1]).GetAngle()
        angle = min(angle, abs(360.0 - angle))
        if error > 1e-6 or angle > 1e-3:
            raise ValueError(f"Authored zero pose does not close: {prim.GetName()}: {error} m, {angle} deg")
        joint_report.append(dict(name=prim.GetName(), type=prim.GetTypeName(),
            body0=joint.GetBody0Rel().GetTargets()[0].name, body1=joint.GetBody1Rel().GetTargets()[0].name,
            excluded=bool(joint.GetExcludeFromArticulationAttr().Get()),
            anchor_error_m=error, frame_error_deg=angle))

    wheel_heights, radii = [], []
    for side in ("left", "right"):
        wheel = stage.GetPrimAtPath(root.GetPath().AppendChild(f"{side}_wheel_link"))
        colliders = [p for p in Usd.PrimRange(wheel) if p.HasAPI(UsdPhysics.CollisionAPI)]
        if len(colliders) != 1 or not colliders[0].IsA(UsdGeom.Cylinder):
            raise ValueError("Expected one cylindrical wheel collider")
        collider = colliders[0]
        cylinder = UsdGeom.Cylinder(collider)
        if cylinder.GetAxisAttr().Get() != "Z":
            raise ValueError("Unexpected wheel cylinder axis")
        transform = cache.GetLocalToWorldTransform(collider)
        axis = transform.TransformDir(Gf.Vec3d(0, 0, 1))
        if abs(axis[2]) > 1e-6 or abs(axis.GetLength() - 1) > 1e-6:
            raise ValueError("Expected horizontal, unit-scale wheel cylinders")
        radius = cylinder.GetRadiusAttr().Get()
        radii.append(radius)
        wheel_heights.append(base_translation[2] - transform.ExtractTranslation()[2] + radius)
    if abs(wheel_heights[0] - wheel_heights[1]) > 1e-6 or abs(radii[0] - radii[1]) > 1e-6:
        raise ValueError("Expected symmetric wheel heights and radii")

    # Flatten to a self-contained asset. Rebase all sibling links, preserving all
    # link-local geometry, joint frames, masses, limits and loop constraints.
    source_stage = stage  # Keep source prim handles alive for the inventory below.
    stage = Usd.Stage.Open(source_stage.Flatten())
    root = stage.GetDefaultPrim()
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            if prim.GetParent() != root:
                raise ValueError("Expected flat rigid-body hierarchy")
            xform = UsdGeom.Xformable(prim)
            matrix = xform.GetLocalTransformation()
            matrix.SetTranslateOnly(matrix.ExtractTranslation() - base_translation)
            xform.MakeMatrixXform().Set(matrix)
        if prim.IsA(UsdPhysics.Joint):
            # Remove CAD position servos (including the stray right_connect_05
            # drive); MotorBank/auxiliary/passive actuators own all effort.
            for drive in ("angular", "linear"):
                if prim.HasAPI(UsdPhysics.DriveAPI, drive):
                    prim.RemoveAPI(UsdPhysics.DriveAPI, drive)
                for prop in list(prim.GetProperties()):
                    if prop.GetName().startswith(f"drive:{drive}:"):
                        prim.RemoveProperty(prop.GetName())
    root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(root.GetPath().AppendChild("base_link")))
    output = ASSETS / "wheelbipe_v2_0.usd"
    stage.GetRootLayer().Export(str(output))

    spec = json.loads((ASSETS / "v1_0.json").read_text())
    spec.update(name="wheelbipe_v2_0", usd=output.name,
        rest_height=round(sum(wheel_heights) / 2, 9), wheel_radius=round(radii[0], 9),
        joint_names=[p.GetName() for p in movable],
        rest_joint_positions={p.GetName(): 0.0 for p in movable},
        termination_bodies=["base_link", "gimbal_yaw_link", "gimbal_pitch_link"],
        spring_joints=["left_spring_2_joint", "right_spring_2_joint"],
        auxiliary_joints=["gimbal_yaw_joint", "gimbal_pitch_joint"],
        passive_joint_patterns=[f"(left|right)_{part}_joint" for part in
            ("leg_2", "leg_3", "leg_4", "leg_5", "connect_03", "connect_05", "connect_30", "connect_50", "spring_1", "spring_3")])
    spec["motors"][0]["joints"] = ["left_leg_1_joint", "left_thigh_joint", "right_leg_1_joint", "right_thigh_joint"]
    spec["motors"][0]["directions"] = [1, 1, -1, -1]
    spec["motors"][1]["directions"] = [1, -1]
    (ASSETS / "v2_0.json").write_text(json.dumps(spec, indent=2) + "\n")
    report = dict(source="model/v2_0.usdz", source_sha256=digest(source),
        asset="src/wheel_legged_lab/assets/" + output.name, asset_sha256=digest(output),
        spec="src/wheel_legged_lab/assets/v2_0.json", spec_sha256=digest(ASSETS / "v2_0.json"),
        base_translation_removed=list(base_translation),
        rigid_bodies=[p.GetName() for p in bodies], dof_count=len(movable),
        loop_constraint_count=len(closures), joints=joint_report,
        rest_height=spec["rest_height"], wheel_radius=spec["wheel_radius"],
        preserved_code_sha256={name: digest(ROOT / "src/wheel_legged_lab" / name) for name in
            ("components/commands.py", "components/jump.py", "components/rewards.py", "ppo.py")},
        validation="USD geometry/constraint inspection only; no simulator or training started")
    (ROOT / "MODEL_MIGRATION.json").write_text(json.dumps(report, indent=2) + "\n")
    provenance_path = ROOT / "ASSET_PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text())
    provenance.update(active_robot=spec["name"], v2_source_asset=report["source"],
        v2_source_sha256=report["source_sha256"], v2_converted_asset=report["asset"],
        v2_converted_sha256=report["asset_sha256"], v2_robot_spec=report["spec"],
        v2_robot_spec_sha256=report["spec_sha256"],
        validation="USD model conversion completed; run check_model.py and static_check.py. No simulator or training started.")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("asset", "dof_count", "loop_constraint_count", "rest_height", "wheel_radius")}, indent=2))


if __name__ == "__main__":
    main()
