"""A sensor module with a sealed 3-way connector header, fixed on its bracket; and the mating harness plug at the end of
its lead, the lead held in a harness clip. All of this project's own generic design, built in code.

Connector frame (both parts, when mated): x toward the latch side, y across the plug's long side, z along the
mating axis, up toward the plug's rear (the plug goes on downward). Dimensions in mm in the comments.

The header (fixed): a cast-aluminium sensor module (44 x 52 x 62) on a steel bracket plate, and on its top the
header: a black shroud (cavity 14.5 x 22.5, 17 deep, 1.8 walls, 1 mm x 45 degree lead-in chamfer at the mouth, a
4 mm keyway in its -y wall) with three pins. On the shroud's +x wall outside, the lock ramp: 2.0 proud, a 30 degree
lead-in face above a flat catch face 9.0 above the cavity floor. In the cavity, the interface seal as a plate on a
sprung joint: the plug's face meets it 6.0 above the floor and pushes it 5.0 down to the seated position, against
the seal's compression (1.5 N/mm, 7.5 N seated); drawn back, the plate follows the plug up. The terminals' and the
seal's drag is the plug's radial seal pad (below). Beside the module a relay box (50 x 38 x 105), behind it a 28 mm
coolant hose at 48 to 76 above the table: the header is reached from above and from the robot's side only.

The plug (an articulation with the lead and the clip): a black PA66 housing, 14 x 22 entering the shroud, a collar
18.4 x 24.8 from 16 to 30 above its face that stops on the shroud's rim (seated: the face 1.0 above the cavity
floor), a key rib on its -y face, a 90 degree wire-dress cover on its -x side from which the lead leaves. On its +x
side the primary latch: a latch arm (two rails and a hook crossbar at its lower end, the hook's lower face a 30
degree lead-in), sprung toward the housing at 15 N/mm (a stand-in for the plastic arm's bending), which the lock
ramp spreads 1.35 mm as the plug goes on, and which snaps in under the catch face once the plug is within 0.6 mm
of home (seated, the hook sits 0.6 below the catch face). Outside the
latch arm, the CPA (connector position assurance) slider, orange, 4 mm proud of the plug's top in its pre-lock
position: pushed down 4.0 mm to home, it lies outside the latch arm and keeps it from spreading. The CPA rests in
pre-lock and in home by a detent (15 N to move it from either); it is held in pre-lock until the plug is seated and
latched (on the real part, a rib on the header releases the CPA's pre-lock finger only when the plug is home), and
with the latch arm spread its lower end lands on the arm's top, so it cannot go home on a half-mated plug.

The radial seal pad: on the plug's -x face, 8 to 15 above its face, on a slide sprung outward (12 N/mm, 1.05 mm
preload against its limit, standing 0.6 proud of the core, its lower edge chamfered): inside the cavity it presses
the plug against the far wall, so contact friction (0.35, both walls) holds the plug against moving either way.
Measured with force_test.py: the plug latches at 32.9 N of steady axial push and stays half-mated at 25 N.

The lead: 190 mm of 7 mm convoluted tube (three wires) from the plug's dress cover to the clip, as four 47.5 mm
links on two-axis bend joints (0.03 N m/rad), with a twist joint at the plug. The clip, on a steel post, also holds
the harness trunk (12 mm, taped) running across it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import tb_parts as tp
from yam_arena import task_file

MM = 1e-3
HEADER = "sensor_header"
PLUG = "harness_plug"

# -- the header (header part frame: origin on the table under the shroud's centre, connector axes) ------------------
FLOOR = 66.0  # the cavity floor above the table
CAV = (7.25, 11.25)  # the cavity's half sizes (x, y): 0.25 mm play round the plug's 7.0 x 11.0
WALL = 1.8
SHROUD_H = 17.0  # rim above the floor
CHAMFER = 1.0
KEYWAY = 2.0  # half width of the keyway in the -y wall
RAMP = (9.0, 2.0, 30.0)  # catch face height above the floor, how far it stands out of the wall, lead-in angle
RAMP_Y = 3.5  # half width
SEAL_TOP = 6.0  # the seal plate's top above the floor, free
SEAL_TRAVEL = 5.0
SEAL_SPRING = 1500.0  # N/m: the seal's compression (7.5 N at the seated position)
MODULE = (22.0, 26.0)  # the module body's half sizes
PLATE = (35.0, 45.0, 4.0)  # the bracket plate's half sizes (x, y) and thickness
RELAY = (25.0, -85.0, -47.0, 105.0)  # the relay box beside it: half size along x, from y to y, height
HOSE = (-48.0, 62.0, 14.0, 70.0)  # the coolant hose behind it (28 mm, along y): x, height of its axis, radius, half length

# -- the plug (plug frame: origin at the middle of its mating face, connector axes) ---------------------------------
CORE = (7.0, 11.0)  # half sizes of the part that enters the shroud
NOSE = (6.0, 10.0, 1.2)  # the stepped nose at the face (half sizes, height)
COLLAR = (9.2, 12.4, 16.0, 30.0)  # half sizes, from z to z
SEAT = 1.0  # seated: the face this far above the cavity floor (the collar on the rim)
RIB = (1.5, 12.5, 3.0, 16.0)  # key rib on the -y face: half width, out to y = -12.5, from z to z
HOOD = (9.2, 12.0, 5.0, 22.4, 30.0)  # the latch's hood: x0, x1, half width, z0, z1
ARM = (10.0, 12.0, 5.0, 4.2, 22.0)  # latch arm rails: x0, x1, outer and inner half width (the ramp passes between), top
HOOK_TOP = 7.4  # the hook's catch face (seated: 0.6 below the ramp's catch face)
HOOK_TIP = 9.7  # the hook's inner face, at rest (0.65 outside the shroud wall)
HOOK_FOOT = 6.8  # the hook's inner face runs from HOOK_TOP down to here, then the 30 degree lead-in
LATCH_SPRING = 15000.0  # N/m, toward the housing
LATCH_TRAVEL = 2.5
CPA = (12.2, 15.2, 5.0, 22.4, 34.0)  # the CPA in pre-lock: x0, x1, half width, z0, z1
CPA_TRAVEL = 4.0
# the radial seal on the plug's -x face: a pad on a sprung slide, pressing the plug against the cavity's walls, so its
# friction holds the plug both ways (the terminals' contact springs and the seal's drag on the real part)
PAD = (8.0, 15.0, 8.0, 7.6, 6.4)  # z0, z1, half width (y), its outer face's -x (free), its inner face's -x
PAD_SPRING = 12000.0  # N/m, outward
PAD_PRELOAD = 1.05  # mm: the drive's target this far outside its free position (the slide's limit)
PAD_TRAVEL = 1.0
CPA_DETENT = 7500.0  # N/m: 15 N at the detent's crest, 2.0 mm from either rest
DRESS = (-13.0, -9.2, 6.0, 17.0, 30.0)  # the wire-dress cover: x0, x1, half width, z0, z1
LEAD_Z = 24.0  # the lead leaves the dress cover here, along -x
LEAD_LEN = 190.0
LINKS = 4
LEAD_R = 3.5
BEND = (0.03, 0.001)  # N m/rad, N m s/rad
CLIP_X = DRESS[0] - LEAD_LEN  # the clip, along the lead

PLUG_MASS = 0.014  # housing with terminals, seals and the dress cover
LATCH_MASS = 0.0005
CPA_MASS = 0.001
LINK_MASS = LEAD_LEN / LINKS * MM * 0.045  # 45 g/m

LOOKS = {
    "housing": ((0.10, 0.10, 0.11), 0.6, 0.0),
    "cpa": ((0.95, 0.45, 0.08), 0.5, 0.0),
    "shroud": ((0.08, 0.08, 0.09), 0.6, 0.0),
    "module": ((0.62, 0.64, 0.66), 0.45, 0.6),
    "bracket": ((0.45, 0.47, 0.50), 0.4, 0.8),
    "pin": ((0.85, 0.72, 0.35), 0.3, 0.9),
    "seal": ((0.20, 0.45, 0.75), 0.7, 0.0),
    "lead": ((0.06, 0.06, 0.06), 0.8, 0.0),
    "tape": ((0.12, 0.12, 0.13), 0.9, 0.0),
    "clip": ((0.92, 0.92, 0.88), 0.6, 0.0),
    "hole": ((0.02, 0.02, 0.02), 0.9, 0.0),
    "relay": ((0.16, 0.17, 0.19), 0.7, 0.0),
    "hose": ((0.05, 0.05, 0.05), 0.85, 0.0),
}
MATERIALS = {"pa66": (0.35, 0.35, "average")}


def _b(body, look, mat, x0, x1, y0, y1, z0, z1) -> tp.Shape:
    return tp._box(body, look, mat, x0 * MM, x1 * MM, y0 * MM, y1 * MM, z0 * MM, z1 * MM)


def _py(body, look, mat, profile, y0, y1) -> tp.Shape:
    """A prism along y: ``profile`` in (x, z), mm."""
    return tp._prism(body, look, mat, [(x * MM, z * MM) for x, z in profile], y0 * MM, y1 * MM)


def _px(body, look, mat, profile, x0, x1) -> tp.Shape:
    """A prism along x: ``profile`` in (y, z), mm."""
    return tp.Shape(body, look, mat, profile=tuple((y * MM, z * MM) for y, z in profile), span=(x0 * MM, x1 * MM),
                    axis="X")


def _hose() -> tp.Shape:
    """The hose: a 16-sided section, along y."""
    x, z, r, half = HOSE
    pts = [(x + r * math.cos(2 * math.pi * i / 16), z + r * math.sin(2 * math.pi * i / 16)) for i in range(16)]
    return _py("header", "hose", "rubber", pts, -half, half)


def header_shapes() -> list[tp.Shape]:
    f, h, w, c = FLOOR, FLOOR + SHROUD_H, WALL, CHAMFER
    cx, cy = CAV
    out = [
        _b("header", "bracket", "steel", -PLATE[0], PLATE[0], -PLATE[1], PLATE[1], 0.0, PLATE[2]),
        _b("header", "module", "steel", -MODULE[0], MODULE[0], -MODULE[1], MODULE[1], PLATE[2], f),
        # the shroud's walls, each with the lead-in chamfer on its inner top edge
        _py("header", "shroud", "pa66", [(cx, f), (cx + w, f), (cx + w, h), (cx + c, h), (cx, h - c)],
            -cy - w, cy + w),
        _py("header", "shroud", "pa66", [(-cx - w, f), (-cx, f), (-cx, h - c), (-cx - c, h), (-cx - w, h)],
            -cy - w, cy + w),
        _px("header", "shroud", "pa66", [(cy, f), (cy + w, f), (cy + w, h), (cy + c, h), (cy, h - c)], -cx, cx),
        # the -y wall, split by the keyway
        _px("header", "shroud", "pa66", [(-cy - w, f), (-cy, f), (-cy, h - c), (-cy - c, h), (-cy - w, h)],
            -cx, -KEYWAY),
        _px("header", "shroud", "pa66", [(-cy - w, f), (-cy, f), (-cy, h - c), (-cy - c, h), (-cy - w, h)],
            KEYWAY, cx),
        # the lock ramp on the +x wall: its flat catch face below, its 30 degree lead-in face above
        _py("header", "shroud", "latch", [(cx + w, f + RAMP[0]), (cx + w + RAMP[1], f + RAMP[0]),
                                           (cx + w, f + RAMP[0] + RAMP[1] / math.tan(math.radians(RAMP[2])))],
            -RAMP_Y, RAMP_Y),
        # neighbours in the installed position: a relay box beside the module, a coolant hose behind it
        _b("header", "relay", "pa66", -RELAY[0], RELAY[0], RELAY[1], RELAY[2], 0.0, RELAY[3]),
        _hose(),
        # the seal plate (sprung)
        _b("seal", "seal", "pa66", -cx + 0.6, cx - 0.6, -cy + 0.6, cy - 0.6, f + SEAL_TOP - 2.0, f + SEAL_TOP),
    ]
    # looks: the three pins, the mounting bolts
    for y in (-6.5, 0.0, 6.5):
        out.append(tp.Shape("header", "pin", None, (0.0, y * MM, (f + 7.0) * MM), radius=0.8 * MM, length=14 * MM))
    for x, y in ((-27, -37), (27, -37), (-27, 37), (27, 37)):
        out.append(tp.Shape("header", "bracket", None, (x * MM, y * MM, (PLATE[2] + 1.5) * MM), radius=4.5 * MM,
                            length=3 * MM))
    return out


def _seg_x(k: int) -> tuple[float, float]:
    """The lead's k-th link (1: at the plug), its x extent (mm)."""
    step = LEAD_LEN / LINKS
    return DRESS[0] - k * step, DRESS[0] - (k - 1) * step


def joint_points() -> list[tuple[float, float, float]]:
    """Where the lead bends: P0 at the dress cover ... P4 at the clip (m)."""
    step = LEAD_LEN / LINKS
    return [((DRESS[0] - k * step) * MM, 0.0, LEAD_Z * MM) for k in range(LINKS + 1)]


def plug_shapes() -> list[tp.Shape]:
    cx, cy = CORE
    hx0, hx1, hw, hz0, hz1 = HOOD
    ax0, ax1, aw, ai, atop = ARM
    k0, k1, kw, kz0, kz1 = CPA
    dx0, dx1, dw, dz0, dz1 = DRESS
    run = (HOOK_FOOT - 4.23)  # the hook's lead-in face: 30 degrees from the mating axis
    foot_x = HOOK_TIP + run * math.tan(math.radians(30.0))
    out = [
        _b("plug", "housing", "pa66", -NOSE[0], NOSE[0], -NOSE[1], NOSE[1], 0.0, NOSE[2]),
        _b("plug", "housing", "pa66", -cx, cx, -cy, cy, NOSE[2], COLLAR[2]),
        _b("plug", "housing", "pa66", -COLLAR[0], COLLAR[0], -COLLAR[1], COLLAR[1], COLLAR[2], COLLAR[3]),
        _b("plug", "housing", "pa66", -RIB[0], RIB[0], -RIB[1], -cy, RIB[2], RIB[3]),
        _b("plug", "housing", "pa66", hx0, hx1, -hw, hw, hz0, hz1),
        _b("plug", "housing", "pa66", dx0, dx1, -dw, dw, dz0, dz1),
        # the latch arm: two rails, the hook crossbar at their lower end, a crossbar at the top
        _b("latch", "housing", "pa66", ax0, ax1, ai, aw, 4.23, atop),
        _b("latch", "housing", "pa66", ax0, ax1, -aw, -ai, 4.23, atop),
        _b("latch", "housing", "pa66", ax0, ax1, -aw, aw, 19.0, atop),
        _py("latch", "housing", "latch", [(HOOK_TIP, HOOK_TOP), (HOOK_TIP, HOOK_FOOT), (foot_x, 4.23), (ax1, 4.23),
                                          (ax1, HOOK_TOP)], -aw, aw),
        # the CPA slider
        _b("cpa", "cpa", "pa66", k0, k1, -kw, kw, kz0, kz1),
        # the radial seal pad, its lower edge chamfered so the shroud's mouth presses it in
        _py("pad", "seal", "pa66", [(-PAD[4], PAD[0]), (-CORE[0], PAD[0]), (-PAD[3], PAD[0] + 1.2), (-PAD[3], PAD[1]),
                                    (-PAD[4], PAD[1])], -PAD[2], PAD[2]),
    ]
    # looks: grip ribs on the collar's sides, the terminal cavities in the face
    for z in (20.0, 23.0, 26.0):
        for s in (-1, 1):
            out.append(_b("plug", "hole", None, -7.0, 7.0, s * COLLAR[1] - 0.15, s * COLLAR[1] + 0.15, z, z + 1.0))
    for y in (-6.5, 0.0, 6.5):
        out.append(_b("plug", "hole", None, -1.2, 1.2, y - 1.2, y + 1.2, -0.1, 0.0))
    # the lead: four links of convoluted tube (bands for its ribs)
    for k in range(1, LINKS + 1):
        x0, x1 = _seg_x(k)
        out.append(_b(f"seg{k}", "lead", "pa66", x0 + 0.3, x1 - 0.3, -LEAD_R, LEAD_R, LEAD_Z - LEAD_R, LEAD_Z + LEAD_R))
    # the clip on its post, and the harness trunk through it
    out += [
        _b("clip", "bracket", "steel", CLIP_X - 12, CLIP_X + 12, -12, 12, 0.0, 15.0),
        _b("clip", "clip", "pa66", CLIP_X - 6, CLIP_X + 4, -8, 8, 15.0, 33.0),
        _b("clip", "tape", "pa66", CLIP_X - 22, CLIP_X - 8, -90, 90, LEAD_Z - 6, LEAD_Z + 6),
    ]
    return out


# -- USD ----------------------------------------------------------------------------------------------------------------
def _joint(stage, paths, name, kind, parent, child, axis, lo, hi, at, stiffness, damping, armature):
    from pxr import Gf, Sdf, UsdPhysics

    j = (UsdPhysics.RevoluteJoint if kind == "angular" else UsdPhysics.PrismaticJoint).Define(stage,
                                                                                            f"/Part/joints/{name}")
    j.CreateBody0Rel().SetTargets([Sdf.Path(paths[parent])])
    j.CreateBody1Rel().SetTargets([Sdf.Path(paths[child])])
    j.CreateAxisAttr(axis)
    j.CreateLowerLimitAttr(lo)
    j.CreateUpperLimitAttr(hi)
    j.CreateLocalPos0Attr(Gf.Vec3f(*at))
    j.CreateLocalPos1Attr(Gf.Vec3f(*at))
    j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), kind)
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(stiffness)
    drive.CreateDampingAttr(damping)
    drive.CreateTargetPositionAttr(0.0)
    tp._apply(j.GetPrim(), "PhysxJointAPI", {"physxJoint:armature": armature})


def _bodies(stage, masses: dict, tiny=()) -> dict:
    from pxr import Gf, UsdPhysics

    paths = {}
    for body, mass in masses.items():
        p = f"/Part/{body}"
        prim = tp._xform(stage, p)
        paths[body] = p
        UsdPhysics.RigidBodyAPI.Apply(prim)
        m = UsdPhysics.MassAPI.Apply(prim)
        m.CreateMassAttr(mass)
        if body in tiny:
            m.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-8, 1e-8, 1e-8))
        tp._apply(prim, "PhysxRigidBodyAPI", {
            "physxRigidBody:solverPositionIterationCount": 32,
            "physxRigidBody:solverVelocityIterationCount": 8,
            "physxRigidBody:maxDepenetrationVelocity": 0.3,
        })
    return paths


def _root(stage, path: str) -> None:
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(path)
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    tp._apply(prim, "PhysxArticulationAPI", {
        "physxArticulation:enabledSelfCollisions": False,
        "physxArticulation:solverPositionIterationCount": 32,
        "physxArticulation:solverVelocityIterationCount": 8,
    })


def write_header_usd(path) -> None:
    stage, root, looks, mats = tp._new_stage(path)
    paths = _bodies(stage, {"header": 2.0, "seal": 0.002})
    tp._write_shapes(stage, HEADER, paths, looks, mats)
    _root(stage, paths["header"])
    _joint(stage, paths, "seal", "linear", "header", "seal", "Z", -SEAL_TRAVEL * MM, 0.0, (0.0, 0.0, FLOOR * MM),
           SEAL_SPRING, 20.0, 0.05)
    stage.GetRootLayer().Save()


BEND_JOINTS = [f"bend_{a}{k}" for k in range(LINKS, -1, -1) for a in ("z", "y")] + ["twist"]


def _plug_bodies() -> dict:
    masses = {"clip": 1.0}
    for k in range(LINKS, 0, -1):
        masses[f"d{k}"] = 2e-4
        masses[f"seg{k}"] = LINK_MASS
    masses.update({"d0": 2e-4, "e0": 2e-4, "plug": PLUG_MASS, "latch": LATCH_MASS, "cpa": CPA_MASS, "pad": 5e-4})
    return masses


def write_plug_usd(path) -> None:
    stage, root, looks, mats = tp._new_stage(path)
    masses = _plug_bodies()
    paths = _bodies(stage, masses, tiny=[b for b in masses if b[0] in "de"])
    tp._write_shapes(stage, PLUG, paths, looks, mats)
    _root(stage, paths["clip"])
    pts = joint_points()
    k_deg, c_deg = math.radians(BEND[0]), math.radians(BEND[1])  # USD angular drives are per degree
    for k in range(LINKS, -1, -1):
        parent = "clip" if k == LINKS else f"seg{k + 1}"
        child = f"seg{k}" if k > 0 else "e0"
        _joint(stage, paths, f"bend_z{k}", "angular", parent, f"d{k}", "Z", -120.0, 120.0, pts[k], k_deg, c_deg, 1e-5)
        _joint(stage, paths, f"bend_y{k}", "angular", f"d{k}", child, "Y", -120.0, 120.0, pts[k], k_deg, c_deg, 1e-5)
    _joint(stage, paths, "twist", "angular", "e0", "plug", "X", -150.0, 150.0, pts[0], k_deg, c_deg, 1e-5)
    _joint(stage, paths, "latch", "linear", "plug", "latch", "X", 0.0, LATCH_TRAVEL * MM,
           (ARM[0] * MM, 0.0, 10 * MM), LATCH_SPRING, 20.0, 0.05)
    _joint(stage, paths, "cpa", "linear", "plug", "cpa", "Z", -CPA_TRAVEL * MM, 0.0, (CPA[0] * MM, 0.0, 28 * MM),
           CPA_DETENT, 20.0, 0.05)
    _joint(stage, paths, "pad", "linear", "plug", "pad", "X", 0.0, PAD_TRAVEL * MM, (-CORE[0] * MM, 0.0, 11.5 * MM),
           PAD_SPRING, 20.0, 0.05)
    stage.GetRootLayer().Save()


# -- arena objects ------------------------------------------------------------------------------------------------------
def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def cpa_lower(art) -> float:
    """The CPA joint's lower limit now (m; -CPA_TRAVEL mm: free to go home)."""
    names = list(art.joint_names)
    lim = art.data.joint_pos_limits.torch[0].detach().cpu().numpy()
    return float(lim[names.index("cpa"), 0])


def set_cpa_lower(art, lower: float) -> None:
    import torch

    names = list(art.joint_names)
    i = names.index("cpa")
    lim = torch.tensor([[[lower, 0.0]]], dtype=torch.float32, device=art.device)
    art.write_joint_position_limit_to_sim_index(limits=lim, joint_ids=[i], warn_limit_violation=False)


def set_cpa_target(art, target: float) -> None:
    import torch

    names = list(art.joint_names)
    t = torch.zeros((art.num_instances, art.num_joints), device=art.device)
    t[:, names.index("cpa")] = target
    t[:, names.index("pad")] = -PAD_PRELOAD * MM  # the seal pad's preload (outward, against its limit)
    art.set_joint_position_target(t)


def header_object(spec: tp.Part):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    from isaaclab_physx.assets.articulation import Articulation as PhysxArticulation

    class _HeaderArticulation(PhysxArticulation):
        def reset(self, env_ids=None, env_mask=None):
            import torch

            super().reset(env_ids, env_mask)
            q = torch.zeros((self.num_instances, self.num_joints), device=self.device)
            self.write_joint_position_to_sim_index(position=q)
            self.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))

    class _Header(Object):
        def _generate_articulation_cfg(self):
            cfg = super()._generate_articulation_cfg()
            cfg.class_type = _HeaderArticulation
            cfg.actuators = {"seal": ImplicitActuatorCfg(joint_names_expr=["seal"], stiffness=SEAL_SPRING,
                                                         damping=20.0, armature=0.05)}
            cfg.init_state.joint_pos = {"seal": 0.0}
            return cfg

    cfg = sim_utils.UsdFileCfg(usd_path=tp.usd_file(HEADER), fix_root_link=True)
    pose = Pose(position_xyz=tuple(spec.pos), rotation_xyzw=_yaw_quat(spec.yaw))
    return _Header(name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=cfg, initial_pose=pose)


def plug_object(spec: tp.Part):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    from isaaclab_physx.assets.articulation import Articulation as PhysxArticulation

    class _PlugArticulation(PhysxArticulation):
        def reset(self, env_ids=None, env_mask=None):
            import torch

            super().reset(env_ids, env_mask)
            q = torch.zeros((self.num_instances, self.num_joints), device=self.device)
            self.write_joint_position_to_sim_index(position=q)
            self.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
            set_cpa_lower(self, 0.0)
            set_cpa_target(self, 0.0)
            _CPA_TARGET[id(self)] = 0.0

        def set_joint_position_target_index(self, *, target, joint_ids=None, env_ids=None, full_data=False):
            super().set_joint_position_target_index(target=target, joint_ids=joint_ids, env_ids=env_ids,
                                                    full_data=full_data)

    class _Plug(Object):
        def _generate_articulation_cfg(self):
            cfg = super()._generate_articulation_cfg()
            cfg.class_type = _PlugArticulation
            cfg.actuators = {
                "lead": ImplicitActuatorCfg(joint_names_expr=["bend_.*", "twist"], stiffness=BEND[0],
                                            damping=BEND[1], armature=1e-5),
                "latch": ImplicitActuatorCfg(joint_names_expr=["latch"], stiffness=LATCH_SPRING, damping=20.0,
                                             armature=0.05),
                "cpa": ImplicitActuatorCfg(joint_names_expr=["cpa"], stiffness=CPA_DETENT, damping=20.0,
                                           armature=0.05),
                "pad": ImplicitActuatorCfg(joint_names_expr=["pad"], stiffness=PAD_SPRING, damping=20.0,
                                           armature=0.05),
            }
            cfg.init_state.joint_pos = {".*": 0.0}
            return cfg

    cfg = sim_utils.UsdFileCfg(usd_path=tp.usd_file(PLUG), fix_root_link=True)
    pose = Pose(position_xyz=tuple(spec.pos), rotation_xyzw=_yaw_quat(spec.yaw))
    return _Plug(name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=cfg, initial_pose=pose)


_CPA_TARGET: dict = {}


def update_cpa(art, open_: bool) -> None:
    """Once per control step: the CPA's pre-lock (held there unless ``open_``: the plug seated and latched) and its
    detent (it rests in pre-lock or in home, whichever side of the crest, half way, it is on)."""
    names = list(art.joint_names)
    q = float(art.data.joint_pos.torch[0, names.index("cpa")].item())
    lower = cpa_lower(art)
    want = -CPA_TRAVEL * MM if open_ else min(max(q, -CPA_TRAVEL * MM), 0.0)
    if open_ and lower > -CPA_TRAVEL * MM + 1e-6 or not open_ and abs(lower - want) > 0.05 * MM and want > lower:
        set_cpa_lower(art, want)
    target = -CPA_TRAVEL * MM if q <= -CPA_TRAVEL * MM / 2 else 0.0
    if _CPA_TARGET.get(id(art)) != target:
        set_cpa_target(art, target)
        _CPA_TARGET[id(art)] = target


def follow_plug(art, face_mm: float) -> None:
    """Once per control step: the terminals' friction acts both ways, so when the plug is drawn back off the seal
    plate the plate comes back up with it (to its free height at most)."""
    import torch

    names = list(art.joint_names)
    i = names.index("seal")
    q = float(art.data.joint_pos.torch[0, i].item())
    top = SEAL_TOP + q / MM  # the plate's top above the floor (mm)
    if face_mm > top + 0.2 and q < 0.0:
        new = min(0.0, (face_mm - SEAL_TOP) * MM)
        pos = art.data.joint_pos.torch.clone()
        pos[:, i] = new
        art.write_joint_position_to_sim_index(position=pos)
        art.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(pos))


@dataclass(frozen=True)
class HeaderState(task_file.ObjectState):
    seal_mm: float = 0.0  # how far the seal plate is pressed down


@dataclass(frozen=True)
class PlugState(task_file.ObjectState):
    """The plug: ``origin`` and ``rot`` of its frame (the middle of its mating face), how far its latch arm is spread
    (mm), how far its CPA is pushed down from pre-lock (mm, home at 4.0)."""

    origin: tuple = (0.0, 0.0, 0.0)
    rot: tuple = ()
    latch_mm: float = 0.0
    cpa_mm: float = 0.0


def _world(base, art, body: str):
    origin = base.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
    i = list(art.body_names).index(body)
    pose = art.data.body_link_pose_w.torch[0, i].detach().cpu().numpy().astype(np.float64)
    from yam_arena.urdf import quat_xyzw_to_matrix

    return pose[:3] - origin, pose[3:], quat_xyzw_to_matrix(pose[3:])


def _joints(art) -> dict:
    q = art.data.joint_pos.torch[0].detach().cpu().numpy().astype(np.float64)
    return {n: float(q[i]) for i, n in enumerate(art.joint_names)}


def header_state(base, name: str) -> HeaderState:
    art = base.scene.articulations[name]
    pos, quat, rot = _world(base, art, "header")
    lo, hi = tp.bounds(HEADER)
    corners = np.array([[(lo, hi)[(i >> k) & 1][k] for k in range(3)] for i in range(8)]) @ rot.T + pos
    return HeaderState(pos=pos + rot @ np.array(tp.local_center(HEADER)), quat_xyzw=quat, lin_vel=np.zeros(3),
                       ang_vel=np.zeros(3), low=corners.min(0), high=corners.max(0),
                       seal_mm=-_joints(art)["seal"] / MM)


def plug_state(base, name: str) -> PlugState:
    art = base.scene.articulations[name]
    pos, quat, rot = _world(base, art, "plug")
    j = _joints(art)
    lo = np.array([-COLLAR[0], -COLLAR[1], 0.0]) * MM
    hi = np.array([COLLAR[0], COLLAR[1], COLLAR[3]]) * MM
    corners = np.array([[(lo, hi)[(i >> k) & 1][k] for k in range(3)] for i in range(8)]) @ rot.T + pos
    return PlugState(pos=pos + rot @ ((lo + hi) / 2), quat_xyzw=quat, lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                     low=corners.min(0), high=corners.max(0), origin=tuple(float(v) for v in pos),
                     rot=tuple(tuple(float(v) for v in r) for r in rot), latch_mm=j["latch"] / MM,
                     cpa_mm=-j["cpa"] / MM)


def install() -> None:
    tp.SHAPES.update({HEADER: header_shapes, PLUG: plug_shapes})
    tp.LOOKS.update(LOOKS)
    tp.MATERIALS.update(MATERIALS)
    tp.CUSTOM_KINDS[HEADER] = (write_header_usd, header_object, header_state)
    tp.CUSTOM_KINDS[PLUG] = (write_plug_usd, plug_object, plug_state)


def part(name: str, kind: str, pos, yaw: float) -> tp.Part:
    lo, hi = tp.bounds(kind)
    size = tuple(b - a for a, b in zip(lo, hi, strict=True))
    return tp.Part(name=name, pos=tuple(float(v) for v in pos), size=size, mass=None, color=LOOKS["housing"][0],
                   kind=kind, yaw=yaw)
