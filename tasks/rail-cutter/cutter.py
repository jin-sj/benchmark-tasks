"""A bench rail cutter: a fixed base (an infeed ledge, a die the rail passes through, an outfeed ledge with a length
scale) and a hinged cutting lever, sprung up.

Frame (the cutter's): origin on the table at the blade's plane, under the middle of the die; y along the rail's
path (+y the infeed side, the robot's left; -y the outfeed side, where the cut length is measured), x across it (+x
away from the arms), z up. The blade's plane is y = 0.

The lever is one hinged body: an arm from the pivot (behind the die, at ``PIVOT``) toward the arms, a handle at its
end, and the blade near the pivot. Its angle is 0 when it is down (a full stroke: the blade through the rail) and
``UP_DEG`` when it rests raised, where a spring holds it. Pulling the handle down swings the blade down through the
die. The blade is a look only: nothing is cut in the simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import tb_parts as tp
from yam_arena import task_file

MM = 1e-3
KIND = "rail_cutter"

# -- geometry (cutter frame, metres) ------------------------------------------------------------------------------
LEDGE_TOP = 40 * MM  # the rail rests on the ledges and the die's floor at this height
LEDGE_HALF_X = 30 * MM
INFEED = (12 * MM, 680 * MM)  # the infeed ledge (a long infeed table), along y: the whole 600 mm rail rests on it
OUTFEED = (-330 * MM, -12 * MM)  # the outfeed ledge, along y
DIE_Y = 12 * MM  # the die spans y -12 .. +12 mm
APERTURE = (37 * MM, 9.5 * MM)  # the die's opening around the rail (35 x 7.5 mm): width, height above the floor
DIE_TOP = LEDGE_TOP + 30 * MM
LEAD_IN = ((1.5 * MM, 3.0 * MM), (3.0 * MM, 2.0 * MM), (4.5 * MM, 1.0 * MM))  # (depth from the mouth, widening)
FENCE = (14 * MM, 3 * MM, 4 * MM)  # the infeed's guide fences, under the rail's lips: inner face from its centreline (the
# rail's base is 27 mm wide: 0.5 mm of play each side), thickness, height
PIVOT = (40 * MM, LEDGE_TOP + 40 * MM)  # the lever's hinge axis (x, z), along y
ARM = (205 * MM, 16 * MM, 12 * MM)  # the lever's arm: length toward the arms from the pivot (+5 mm behind), width, thickness
HANDLE = (80 * MM, 22 * MM)  # the handle at the arm's end: length, square section
UP_DEG = 45.0  # the lever rests raised this far (its upper stop)
CUT_DEG = 4.0  # the blade has gone through the rail when the lever is this far from down
TOUCH_DEG = 20.0  # the blade reaches the rail's top at about this angle
SCALE_TO = 300 * MM  # the scale's marks run from the blade to here along the outfeed (-y)
LEVER_MASS = 0.3
SPRING = 2.0  # N m/rad, its target beyond the upper stop: about 10 N at the handle to hold the lever down
PRELOAD_DEG = 15.0
DAMPING = 0.05
ARMATURE = 1e-4

LOOKS = {"cutter": ((0.20, 0.36, 0.62), 0.5, 0.2), "grip": ((0.85, 0.15, 0.10), 0.6, 0.0),
         "blade": ((0.80, 0.82, 0.85), 0.25, 0.9), "mark": ((0.05, 0.05, 0.06), 0.8, 0.0)}


def handle_point(deg: float) -> np.ndarray:
    """The middle of the handle (x, z) in the cutter frame with the lever at ``deg``."""
    a = math.radians(deg)
    r = ARM[0] - 5 * MM - HANDLE[0] / 2  # from the pivot to the handle's middle
    return np.array([PIVOT[0] - r * math.cos(a), PIVOT[1] + r * math.sin(a)])


def cutter_shapes() -> list[tp.Shape]:
    b = lambda *v, look="cutter", mat="steel": tp._box("base", look, mat, *v)  # noqa: E731
    hx, ax, ah = LEDGE_HALF_X, APERTURE[0] / 2, LEDGE_TOP + APERTURE[1]
    out = [
        b(-hx, hx, *INFEED, 0.0, LEDGE_TOP), b(-hx, hx, *OUTFEED, 0.0, LEDGE_TOP),  # the ledges
        b(-ax, ax, -DIE_Y, DIE_Y, 0.0, LEDGE_TOP),  # the die's floor
        b(-hx - 1 * MM, -ax, -DIE_Y, DIE_Y - LEAD_IN[-1][0], 0.0, DIE_TOP),  # the cheeks
        b(ax, hx + 1 * MM, -DIE_Y, DIE_Y - LEAD_IN[-1][0], 0.0, DIE_TOP),
        b(-ax, ax, -DIE_Y, DIE_Y - LEAD_IN[-1][0], ah, DIE_TOP),  # the die's top
        b(hx + 1 * MM, PIVOT[0] + 12 * MM, -DIE_Y, DIE_Y, 0.0, PIVOT[1] + 10 * MM),  # the post carrying the hinge
    ]
    # the guide fences along the infeed ledge, either side of the rail, from the die's mouth to the ledge's end
    fx, ft, fh = FENCE
    out += [b(-fx - ft, -fx, DIE_Y, INFEED[1], LEDGE_TOP, LEDGE_TOP + fh),
            b(fx, fx + ft, DIE_Y, INFEED[1], LEDGE_TOP, LEDGE_TOP + fh)]
    # the die's mouth on the infeed side: a stepped lead-in, the opening widening toward the infeed
    d0 = 0.0
    for d1, grow in LEAD_IN:
        y0, y1, d0 = DIE_Y - d1, DIE_Y - d0, d1
        wx, top = ax + grow, ah + grow
        out += [b(-hx - 1 * MM, -wx, y0, y1, 0.0, DIE_TOP), b(wx, hx + 1 * MM, y0, y1, 0.0, DIE_TOP),
                b(-wx, wx, y0, y1, top, DIE_TOP)]
    # the length scale on the outfeed ledge's top, on the near side of the rail: a mark every 10 mm, longer every 50
    for i in range(int(round(SCALE_TO / (10 * MM))) + 1):
        y = -i * 10 * MM
        long = 10 if i % 10 == 0 else (7 if i % 5 == 0 else 4)
        out.append(b(-hx, -hx + long * MM, y - 0.4 * MM, y + 0.4 * MM, LEDGE_TOP, LEDGE_TOP + 0.2 * MM,
                     look="mark", mat=None))
    # the lever, drawn down (angle 0): arm, handle, blade (a look only)
    px, pz = PIVOT
    lv = lambda *v, look="cutter", mat="steel": tp._box("lever", look, mat, *v)  # noqa: E731
    x_end = px - (ARM[0] - 5 * MM)
    out += [
        lv(x_end, px + 5 * MM, -ARM[1] / 2, ARM[1] / 2, pz - ARM[2] / 2, pz + ARM[2] / 2),
        lv(x_end, x_end + HANDLE[0], -HANDLE[1] / 2, HANDLE[1] / 2, pz - HANDLE[1] / 2, pz + HANDLE[1] / 2,
           look="grip", mat="rubber"),
        tp._box("lever", "blade", None, -ax, px - 8 * MM, -0.75 * MM, 0.75 * MM, LEDGE_TOP - 2 * MM, pz - ARM[2] / 2),
    ]
    return out


def install() -> None:
    tp.SHAPES[KIND] = cutter_shapes
    tp.LOOKS.update(LOOKS)
    tp.CUSTOM_KINDS[KIND] = (write_usd, arena_object, state)


def part(name: str, pos) -> tp.Part:
    lo, hi = tp.bounds(KIND)
    size = tuple(b - a for a, b in zip(lo, hi, strict=True))
    return tp.Part(name=name, pos=tuple(float(v) for v in pos), size=size, mass=None, color=LOOKS["cutter"][0], kind=KIND)


def write_usd(path) -> None:
    from pxr import Gf, Sdf, UsdPhysics

    stage, root, looks, mats = tp._new_stage(path)
    paths = {}
    for body in ("base", "lever"):
        p = f"/Part/{body}"
        prim = tp._xform(stage, p)
        paths[body] = p
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(5.0 if body == "base" else LEVER_MASS)
        tp._apply(prim, "PhysxRigidBodyAPI", {
            "physxRigidBody:solverPositionIterationCount": 32,
            "physxRigidBody:solverVelocityIterationCount": 4,
            "physxRigidBody:maxDepenetrationVelocity": 0.3,
        })
    tp._write_shapes(stage, KIND, paths, looks, mats)
    base_prim = stage.GetPrimAtPath(paths["base"])
    UsdPhysics.ArticulationRootAPI.Apply(base_prim)
    tp._apply(base_prim, "PhysxArticulationAPI", {
        "physxArticulation:enabledSelfCollisions": False,
        "physxArticulation:solverPositionIterationCount": 32,
        "physxArticulation:solverVelocityIterationCount": 4,
    })
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Part/joints/lever")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(paths["base"])])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(paths["lever"])])
    joint.CreateAxisAttr("Y")  # about +y a positive angle raises the lever's handle (its -x end)
    joint.CreateLowerLimitAttr(0.0)
    joint.CreateUpperLimitAttr(UP_DEG)
    pivot = Gf.Vec3f(PIVOT[0], 0.0, PIVOT[1])
    joint.CreateLocalPos0Attr(pivot)
    joint.CreateLocalPos1Attr(pivot)
    joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(math.radians(SPRING))
    drive.CreateDampingAttr(math.radians(DAMPING))
    drive.CreateTargetPositionAttr(UP_DEG + PRELOAD_DEG)
    tp._apply(joint.GetPrim(), "PhysxJointAPI", {"physxJoint:armature": ARMATURE})
    stage.GetRootLayer().Save()


def arena_object(spec: tp.Part):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    from isaaclab_physx.assets.articulation import Articulation as PhysxArticulation

    up = math.radians(UP_DEG)
    target = math.radians(UP_DEG + PRELOAD_DEG)

    class _CutterArticulation(PhysxArticulation):
        def _spring(self):
            import torch

            self.set_joint_position_target(torch.full((self.num_instances, self.num_joints), target, device=self.device))

        def reset(self, env_ids=None, env_mask=None):
            # the lever back up, at rest, its spring holding it up
            import torch

            super().reset(env_ids, env_mask)
            q = torch.full((self.num_instances, self.num_joints), up, device=self.device)
            self.write_joint_position_to_sim_index(position=q)
            self.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
            self._spring()

        def set_joint_position_target_index(self, *, target, joint_ids=None, env_ids=None, full_data=False):
            # only reset events write targets this way (they would set the target to the default)
            super().set_joint_position_target_index(target=target, joint_ids=joint_ids, env_ids=env_ids,
                                                    full_data=full_data)
            self._spring()

    class _Cutter(Object):
        def _generate_articulation_cfg(self):
            cfg = super()._generate_articulation_cfg()
            cfg.class_type = _CutterArticulation
            cfg.actuators = {"lever": ImplicitActuatorCfg(joint_names_expr=["lever"], stiffness=SPRING,
                                                          damping=DAMPING, armature=ARMATURE)}
            cfg.init_state.joint_pos = {"lever": up}
            return cfg

    cfg = sim_utils.UsdFileCfg(usd_path=tp.usd_file(KIND), fix_root_link=True)
    pose = Pose(position_xyz=tuple(spec.pos), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    return _Cutter(name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=cfg, initial_pose=pose)


@dataclass(frozen=True)
class CutterState(task_file.ObjectState):
    """The cutter's ``ObjectState`` (its origin) plus its lever's angle in degrees (0 down, 45 raised)."""

    lever_deg: float = UP_DEG


def state(base, name: str) -> CutterState:
    art = base.scene.articulations[name]
    origin = base.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
    pose = art.data.root_pose_w.torch[0].detach().cpu().numpy().astype(np.float64)
    pos = pose[:3] - origin
    q = art.data.joint_pos.torch[0].detach().cpu().numpy().astype(np.float64)
    lo, hi = tp.bounds(KIND)
    return CutterState(pos=pos + np.array(tp.local_center(KIND)), quat_xyzw=pose[3:], lin_vel=np.zeros(3),
                       ang_vel=np.zeros(3), low=pos + np.array(lo), high=pos + np.array(hi),
                       lever_deg=float(math.degrees(q[list(art.joint_names).index("lever")])))
