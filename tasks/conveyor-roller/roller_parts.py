"""A light-duty gravity roller conveyor section, its rollers, and two parts trays.

Conveyor frame: origin on the table on the conveyor's centreline under its near end; x along the conveyor (away from
the arms), y across it (+y the robot's left), z up. Two formed steel angle rails (a 63.5 mm web standing up, a 38.1 mm
leg out along the table, 12 gauge: 2.7 mm), 381 mm between their webs' inner faces (15 in between frames), 610 mm long,
tied at each end by a steel angle. Each web has a row of 7/16 in hex holes (11.9 mm across flats, flats on top
and bottom), 44.5 mm above the table, every 114.3 mm (4.5 in centres): five roller positions.

Roller (roller frame: origin at the middle of the axle, y along the axle, +y its left end as installed, z up): a 1.9 in
(48.3 mm) galvanized steel tube, 368.3 mm long, on two ball bearings, on a 7/16 in (11.1 mm across flats) hex steel
axle 419.1 mm long, which sticks out 25.4 mm past each end of the tube. The axle is spring-retained: it slides through
the bearings toward the roller's left end, up to 16 mm, against a spring (4 N to start it, 13.6 N at full travel), and
springs back; it cannot slide the other way. So pulling the left axle end outward (or pushing the right end in)
shortens the axle's right end by up to 16 mm. The tube turns freely on the axle (the bearings).

Seated, the axle ends stick out through the two hex holes 16.4 mm past the rails' outer faces; the tube's ends are
6.35 mm from the webs. To get the right end out of its hole the axle must slide 12.7 mm or more to the left once the
tube's left end touches the left web (19.05 mm in all from centred).

The roller is one articulation: the tube (root), a hub on the bearings' axis (a revolute joint, the bearings), and the
axle on the hub (a prismatic joint along y with the spring as its drive). The conveyor's frame and its four other
rollers are one fixed collider.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import tb_parts as tp
from yam_arena import task_file

MM = 1e-3
IN = 25.4 * MM

CONVEYOR = "conveyor_section"
ROLLER = "conveyor_roller"
WORN = "worn_roller"
TRAY = "parts_tray"
REJECT = "reject_tray"

# -- the conveyor (conveyor frame) -----------------------------------------------------------------------------------
BF = 15 * IN  # between frames: the webs' inner faces
WEB_T = 2.7 * MM  # 12 gauge steel
FRAME_H = 2.5 * IN  # 63.5 mm
LEG = 1.5 * IN  # the angle's leg along the table, outward
FRAME_LEN = 24 * IN  # 610 mm
PITCH = 4.5 * IN  # roller centres, 114.3 mm
FIRST = 57.15 * MM  # the first roller's centre from the frame's near end
N_POS = 5
HOLE_Z = 44.5 * MM  # the hex holes' centres above the table (the rollers set high: tube tops 5.1 mm above the rails)
HOLE_AF = 11.9 * MM  # hex holes, across flats
INNER_Y = BF / 2  # the webs' inner faces
OUTER_Y = BF / 2 + WEB_T  # ... and outer faces
TIE_W = 25.4 * MM  # the end ties: steel angle, 25.4 x 25.4 mm

# -- the roller (roller frame) ---------------------------------------------------------------------------------------
TUBE_R = 1.9 * IN / 2  # 24.13 mm
TUBE_LEN = 14.5 * IN  # 368.3 mm
TUBE_WALL = 1.5 * MM  # 16 gauge
AXLE_AF = 7 / 16 * IN  # 11.11 mm across flats
AXLE_LEN = 16.5 * IN  # 419.1 mm
STUB = (AXLE_LEN - TUBE_LEN) / 2  # 25.4 mm past each end of the tube
CHAMFER = 0.5 * MM  # 45 degrees, round each axle end's edge
TRAVEL = 16 * MM  # the axle slides this far toward the roller's left end
SPRING_K = 600.0  # N/m (0.6 N/mm) ...
SPRING_PRELOAD = 4.0  # N: 4 N to start, 13.6 N at full travel
TUBE_MASS = 0.80  # kg: tube and both bearings
AXLE_MASS = 0.35  # kg: 419 mm of 7/16 in hex steel
HUB_MASS = 0.01
BEARING_DRAG = 2e-4  # N m s/rad: greased ball bearings, the tube spins on ...
SEAL_K = 0.5  # N m/rad: ... and their seals' drag: the axle keeps its turn in the tube until twisted more than ...
SEAL_SLIP = math.radians(2.0)  # ... this far against it (about 0.017 N m), then turns on
HEX_TWIST_NEW = 15.0  # degrees: the new roller's axle, as it lies in its tray, turned this far from flats on top and bottom

# -- trays (tray frame: origin on the table under its middle; long side along x) --------------------------------------
TRAY_INNER = (460 * MM, 80 * MM)
TRAY_WALL = (25 * MM, 3 * MM)  # height, thickness
TRAY_FLOOR = 3 * MM

LOOKS = {
    "frame": ((0.30, 0.42, 0.62), 0.5, 0.3),  # painted steel, blue
    "galv": ((0.80, 0.82, 0.84), 0.3, 0.9),
    "worn": ((0.50, 0.44, 0.38), 0.8, 0.4),  # dull, rust-stained steel
    "rust": ((0.45, 0.24, 0.12), 0.9, 0.1),
    "scored": ((0.28, 0.27, 0.26), 0.7, 0.5),  # the flat, scored band
    "paint_mark": ((0.85, 0.10, 0.08), 0.6, 0.0),  # the red paint mark on the worn roller
    "axle": ((0.62, 0.63, 0.65), 0.35, 0.9),
    "bearing": ((0.20, 0.20, 0.22), 0.5, 0.5),
    "hole": ((0.05, 0.05, 0.06), 0.9, 0.0),
    "tray_new": ((0.15, 0.30, 0.65), 0.6, 0.0),
    "tray_reject": ((0.35, 0.36, 0.38), 0.7, 0.0),
}
MATERIALS = {"steel": (0.30, 0.30, "average"), "plastic": (0.40, 0.40, "average")}


def roller_xs() -> list[float]:
    """The roller positions' x (conveyor frame)."""
    return [FIRST + k * PITCH for k in range(N_POS)]


# -- shapes ------------------------------------------------------------------------------------------------------------
def _poly(n: int, r: float, phase: float = 0.0) -> list[tuple[float, float]]:
    return [(r * math.cos(phase + 2 * math.pi * i / n), r * math.sin(phase + 2 * math.pi * i / n)) for i in range(n)]


def hex_points(af: float, twist_deg: float = 0.0) -> list[tuple[float, float]]:
    """A hexagon (x, z), ``af`` across flats, flats on top and bottom (turned ``twist_deg`` about y)."""
    r = af / math.sqrt(3.0)
    return _poly(6, r, math.radians(twist_deg))


def web_shapes(body: str, y0: float, y1: float, xs: list[float]) -> list[tp.Shape]:
    """One rail's web (x from 0 to FRAME_LEN, z from 0 to FRAME_H, y from y0 to y1) with a hex hole at each x."""
    m = "steel"
    b = lambda *v: tp._box(body, "frame", m, *v)  # noqa: E731
    af, cr = HOLE_AF / 2, HOLE_AF / math.sqrt(3.0)  # half across flats (up and down), corner radius (along x)
    z0, z1 = HOLE_Z - af, HOLE_Z + af
    out = [b(0.0, FRAME_LEN, y0, y1, 0.0, z0), b(0.0, FRAME_LEN, y0, y1, z1, FRAME_H)]
    edges = [0.0] + [v for x in xs for v in (x - cr, x + cr)] + [FRAME_LEN]
    for a, c in zip(edges[::2], edges[1::2], strict=True):
        out.append(b(a, c, y0, y1, z0, z1))
    for x in xs:
        h = HOLE_Z
        for sx in (-1, 1):
            for sz in (-1, 1):  # the corners of the hole's box outside the hexagon
                tri = [(x + sx * cr, h), (x + sx * cr / 2, h + sz * af), (x + sx * cr, h + sz * af)]
                out.append(tp._prism(body, "frame", m, tri, y0, y1))
        # the hole's dark look, a hair inside the web's faces
        out.append(tp._prism(body, "hole", None, [(x + px, HOLE_Z + pz) for px, pz in hex_points(HOLE_AF)],
                             y0 + 0.2 * MM, y1 - 0.2 * MM))
    return out


def tube_shapes(body: str, look: str, worn: bool) -> list[tp.Shape]:
    """A roller's tube, centred on the roller frame's origin (its collider a 24-sided prism)."""
    hl = TUBE_LEN / 2
    out = [tp.Shape(body, look, "steel", profile=tuple(_poly(24, TUBE_R, math.radians(-0.5))), span=(-hl, hl), visible=False),
           tp.Shape(body, look, None, (0.0, 0.0, 0.0), radius=TUBE_R, length=TUBE_LEN, axis="Y")]
    for s in (-1, 1):  # the bearings' faces in the tube's ends
        y = s * (hl - 0.5 * MM)
        out.append(tp.Shape(body, "bearing", None, (0.0, y + s * 0.6 * MM, 0.0), radius=TUBE_R - TUBE_WALL,
                            length=0.4 * MM, axis="Y"))
    if worn:
        # a flat, scored band round the middle (as it lies seated: on top) and a red paint mark at its left end
        band = lambda look, y0, y1, dr: tp.Shape(body, look, None, (0.0, (y0 + y1) / 2, 0.0),  # noqa: E731
                                                 radius=TUBE_R + dr, length=y1 - y0, axis="Y")
        out += [band("scored", -40 * MM, 25 * MM, 0.15 * MM), band("rust", 70 * MM, 95 * MM, 0.15 * MM),
                band("rust", -130 * MM, -115 * MM, 0.15 * MM), band("paint_mark", hl - 40 * MM, hl - 10 * MM, 0.3 * MM)]
    return out


def axle_shapes(body: str, twist_deg: float = 0.0, collide: bool = True) -> list[tp.Shape]:
    """The hex axle along y, centred on the origin; its ends chamfered (as lofts, see ``LOFTS``)."""
    hl = AXLE_LEN / 2
    return [tp._prism(body, "axle", "steel" if collide else None, hex_points(AXLE_AF, twist_deg), -hl + CHAMFER,
                      hl - CHAMFER)]


def axle_tip_lofts(body: str) -> list[tuple]:
    """The chamfered tips: (body, look, material, outer hex at the chamfer's start, inner hex at the tip, y0, y1)."""
    hl = AXLE_LEN / 2
    big, small = hex_points(AXLE_AF), hex_points(AXLE_AF - 2 * CHAMFER)
    return [(body, "axle", "steel", big, small, hl - CHAMFER, hl), (body, "axle", "steel", big, small,
                                                                      -hl + CHAMFER, -hl)]


def conveyor_shapes() -> list[tp.Shape]:
    """The frame (two angle rails, two end ties) and the four other rollers, fixed. Conveyor frame."""
    body = CONVEYOR
    xs = roller_xs()
    out = []
    for s in (-1, 1):
        y0, y1 = (INNER_Y, OUTER_Y) if s > 0 else (-OUTER_Y, -INNER_Y)
        out += web_shapes(body, y0, y1, xs)
        ly0, ly1 = (INNER_Y, INNER_Y + LEG) if s > 0 else (-INNER_Y - LEG, -INNER_Y)
        out.append(tp._box(body, "frame", "steel", 0.0, FRAME_LEN, ly0, ly1, 0.0, WEB_T))
    for x0 in (0.0, FRAME_LEN - TIE_W):  # end ties: an angle across, between the webs, low
        out.append(tp._box(body, "frame", "steel", x0, x0 + TIE_W, -INNER_Y, INNER_Y, 0.0, WEB_T))
        xv = x0 if x0 == 0.0 else FRAME_LEN - WEB_T
        out.append(tp._box(body, "frame", "steel", xv, xv + WEB_T, -INNER_Y, INNER_Y, 0.0, TIE_W))
    for k, x in enumerate(xs):
        if k == WORN_INDEX:
            continue
        for sh in tube_shapes(body, "galv", False):
            out.append(_moved(sh, (x, 0.0, HOLE_Z)))
        out.append(_moved(axle_shapes(body, collide=False)[0], (x, 0.0, HOLE_Z)))
    return out


WORN_INDEX = 1  # the worn roller's position (the second from the near end)


def _moved(sh: tp.Shape, d) -> tp.Shape:
    from dataclasses import replace

    if sh.profile is not None:  # a prism along y: its profile is (x, z)
        return replace(sh, profile=tuple((x + d[0], z + d[2]) for x, z in sh.profile),
                       span=(sh.span[0] + d[1], sh.span[1] + d[1]))
    return replace(sh, center=tuple(c + v for c, v in zip(sh.center, d, strict=True)))


def roller_shapes(look: str, worn: bool):
    def f() -> list[tp.Shape]:
        return tube_shapes("tube", look, worn) + axle_shapes("axle")
    return f


def tray_shapes(kind: str, look: str):
    def f() -> list[tp.Shape]:
        ix, iy = TRAY_INNER
        h, t = TRAY_WALL
        hx, hy = ix / 2 + t, iy / 2 + t
        b = lambda *v: tp._box(kind, look, "plastic", *v)  # noqa: E731
        return [b(-hx, hx, -hy, hy, 0.0, TRAY_FLOOR), b(-hx, -hx + t, -hy, hy, 0.0, h), b(hx - t, hx, -hy, hy, 0.0, h),
                b(-hx, hx, -hy, -hy + t, 0.0, h), b(-hx, hx, hy - t, hy, 0.0, h)]
    return f


# -- USD: the roller (an articulation) -------------------------------------------------------------------------------
def _loft_mesh(stage, path, a, b, y0, y1):
    """A convex mesh between hexagon ``a`` at y0 and hexagon ``b`` at y1 (both (x, z) lists)."""
    from pxr import Gf, UsdGeom, Vt

    n = len(a)
    pts = [(x, y0, z) for x, z in a] + [(x, y1, z) for x, z in b]
    counts, idx = [], []
    counts.append(n)
    idx += list(range(n))[::-1]
    counts.append(n)
    idx += [n + i for i in range(n)]
    for i in range(n):
        j = (i + 1) % n
        counts.append(4)
        idx += [i, j, n + j, n + i]
    geom = UsdGeom.Mesh.Define(stage, path)
    geom.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    geom.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    geom.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    geom.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    geom.CreateDoubleSidedAttr(True)
    return geom


def write_roller_usd(kind: str):
    def write(path) -> None:
        from pxr import Gf, Sdf, UsdPhysics, UsdShade

        stage, root, looks, mats = tp._new_stage(path)
        paths = {}
        inertia = {"tube": None, "hub": (1e-6, 1e-6, 1e-6), "axle": None}
        for body, mass in (("tube", TUBE_MASS), ("hub", HUB_MASS), ("axle", AXLE_MASS)):
            p = f"/Part/{body}"
            prim = tp._xform(stage, p)
            paths[body] = p
            UsdPhysics.RigidBodyAPI.Apply(prim)
            m = UsdPhysics.MassAPI.Apply(prim)
            m.CreateMassAttr(mass)
            m.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
            if inertia[body] is not None:
                m.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia[body]))
            tp._apply(prim, "PhysxRigidBodyAPI", {
                "physxRigidBody:solverPositionIterationCount": 32,
                "physxRigidBody:solverVelocityIterationCount": 8,
                "physxRigidBody:maxDepenetrationVelocity": 0.3,
            })
        tp._write_shapes(stage, kind, paths, looks, mats)
        for i, (body, look, mat, a, b, y0, y1) in enumerate(axle_tip_lofts("axle")):
            geom = _loft_mesh(stage, f"{paths[body]}/tip_{i}", a, b, y0, y1)
            prim = geom.GetPrim()
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(looks[look])
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
            UsdShade.MaterialBindingAPI(prim).Bind(mats[mat], UsdShade.Tokens.strongerThanDescendants, "physics")
            tp._apply(prim, "PhysxCollisionAPI", {"physxCollision:contactOffset": tp.CONTACT_OFFSET,
                                                   "physxCollision:restOffset": 0.0})
        root_prim = stage.GetPrimAtPath(paths["tube"])
        UsdPhysics.ArticulationRootAPI.Apply(root_prim)
        tp._apply(root_prim, "PhysxArticulationAPI", {
            "physxArticulation:enabledSelfCollisions": False,
            "physxArticulation:solverPositionIterationCount": 32,
            "physxArticulation:solverVelocityIterationCount": 8,
        })
        for name, cls, parent, child, axis, lo, hi, k, d, target in (
            ("bearing", UsdPhysics.RevoluteJoint, "tube", "hub", "Y", None, None, math.radians(SEAL_K), BEARING_DRAG, 0.0),
            ("slide", UsdPhysics.PrismaticJoint, "hub", "axle", "Y", 0.0, TRAVEL, SPRING_K, 2.0,
             -SPRING_PRELOAD / SPRING_K),
        ):
            j = cls.Define(stage, f"/Part/joints/{name}")
            j.CreateBody0Rel().SetTargets([Sdf.Path(paths[parent])])
            j.CreateBody1Rel().SetTargets([Sdf.Path(paths[child])])
            j.CreateAxisAttr(axis)
            if lo is not None:
                j.CreateLowerLimitAttr(lo)
                j.CreateUpperLimitAttr(hi)
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
            j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
            drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular" if name == "bearing" else "linear")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(k)
            drive.CreateDampingAttr(d)
            drive.CreateTargetPositionAttr(target)
            tp._apply(j.GetPrim(), "PhysxJointAPI", {"physxJoint:armature": 1e-4})
        stage.GetRootLayer().Save()
    return write


INIT_TWIST: dict[str, float] = {}  # roller name -> its axle's starting turn about its axis (rad), set by the task
_HELD: dict = {}  # articulation id -> where the bearings' seals hold the axle's turn (rad)


def _targets(art, held: float) -> None:
    import torch

    t = torch.zeros((art.num_instances, art.num_joints), device=art.device)
    names = list(art.joint_names)
    t[:, names.index("slide")] = -SPRING_PRELOAD / SPRING_K
    t[:, names.index("bearing")] = held
    art.set_joint_position_target(t)


def roller_update(base, name: str, out) -> None:
    """The seals' drag (once per control step): the axle's turn is held where it is unless twisted more than
    SEAL_SLIP from there, and then the hold moves with it."""
    art = base.scene.articulations[name]
    q = float(art.data.joint_pos.torch[0][list(art.joint_names).index("bearing")])
    held = _HELD.get(id(art), q)
    held = min(max(held, q - SEAL_SLIP), q + SEAL_SLIP)
    _HELD[id(art)] = held
    _targets(art, held)


def roller_object(spec: tp.Part):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    from isaaclab_physx.assets.articulation import Articulation as PhysxArticulation

    twist = INIT_TWIST.get(spec.name, 0.0)

    class _RollerArticulation(PhysxArticulation):
        def _spring(self):
            _targets(self, _HELD.get(id(self), twist))

        def reset(self, env_ids=None, env_mask=None):
            import torch

            super().reset(env_ids, env_mask)
            q = torch.zeros((self.num_instances, self.num_joints), device=self.device)
            q[:, list(self.joint_names).index("bearing")] = twist
            self.write_joint_position_to_sim_index(position=q)
            self.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
            _HELD[id(self)] = twist
            self._spring()

        def set_joint_position_target_index(self, *, target, joint_ids=None, env_ids=None, full_data=False):
            super().set_joint_position_target_index(target=target, joint_ids=joint_ids, env_ids=env_ids,
                                                    full_data=full_data)
            self._spring()

    class _Roller(Object):
        def _generate_articulation_cfg(self):
            cfg = super()._generate_articulation_cfg()
            cfg.class_type = _RollerArticulation
            cfg.actuators = {
                "bearing": ImplicitActuatorCfg(joint_names_expr=["bearing"], stiffness=SEAL_K, damping=BEARING_DRAG,
                                               armature=1e-4),
                "slide": ImplicitActuatorCfg(joint_names_expr=["slide"], stiffness=SPRING_K, damping=2.0,
                                             armature=1e-4),
            }
            cfg.init_state.joint_pos = {"bearing": twist, "slide": 0.0}
            return cfg

    cfg = sim_utils.UsdFileCfg(usd_path=tp.usd_file(spec.kind), fix_root_link=False)
    c, s = math.cos(spec.yaw / 2), math.sin(spec.yaw / 2)
    pose = Pose(position_xyz=tuple(spec.pos), rotation_xyzw=(0.0, 0.0, s, c))
    return _Roller(name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=cfg, initial_pose=pose)


@dataclass(frozen=True)
class RollerState(task_file.ObjectState):
    """A roller's ``ObjectState`` (pos: the tube's middle) plus: the axle's two ends (world, m: ``tip_left`` the end at
    the roller's +y, ``tip_right`` the other), how far the axle is slid toward the left end (m), and the axle's hex
    turn about its axis (degrees, 0: flats on top and bottom, as the holes are, when the roller lies level)."""

    tip_left: tuple = (0.0, 0.0, 0.0)
    tip_right: tuple = (0.0, 0.0, 0.0)
    slide: float = 0.0
    hex_deg: float = 0.0


def _rot(q) -> np.ndarray:
    from yam_arena.urdf import quat_xyzw_to_matrix

    return quat_xyzw_to_matrix(np.asarray(q, dtype=float))


def roller_state(base, name: str) -> RollerState:
    art = base.scene.articulations[name]
    origin = base.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
    poses = art.data.body_link_pose_w.torch[0].detach().cpu().numpy().astype(np.float64)
    names = list(art.body_names)
    tube = poses[names.index("tube")]
    axle = poses[names.index("axle")]
    q = art.data.joint_pos.torch[0].detach().cpu().numpy().astype(np.float64)
    jn = list(art.joint_names)
    pos = tube[:3] - origin
    rt, ra = _rot(tube[3:]), _rot(axle[3:])
    pa = axle[:3] - origin
    tl = pa + ra @ np.array([0.0, AXLE_LEN / 2, 0.0])
    tr = pa - ra @ np.array([0.0, AXLE_LEN / 2, 0.0])
    lo, hi = tp.bounds(_KIND_OF.get(name, ROLLER))
    corners = np.array([[(lo, hi)[(i >> k) & 1][k] for k in range(3)] for i in range(8)]) @ rt.T + pos
    # the hex's turn: the angle of the axle's local z about its local y, against the world's vertical
    up = ra[:, 2]
    ax = ra[:, 1]
    ref = np.array([0.0, 0.0, 1.0]) - ax * ax[2]
    hex_deg = 0.0
    if np.linalg.norm(ref) > 1e-6:
        ref /= np.linalg.norm(ref)
        hex_deg = math.degrees(math.atan2(float(np.dot(np.cross(ref, up), ax)), float(np.dot(ref, up))))
    return RollerState(pos=pos, quat_xyzw=tube[3:], lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                       low=corners.min(axis=0), high=corners.max(axis=0), tip_left=tuple(float(v) for v in tl),
                       tip_right=tuple(float(v) for v in tr), slide=float(q[jn.index("slide")]), hex_deg=hex_deg)


_KIND_OF: dict[str, str] = {}


def install() -> None:
    tp.SHAPES.update({CONVEYOR: conveyor_shapes, ROLLER: roller_shapes("galv", False), WORN: roller_shapes("worn", True),
                      TRAY: tray_shapes(TRAY, "tray_new"), REJECT: tray_shapes(REJECT, "tray_reject")})
    tp.LOOKS.update(LOOKS)
    tp.MATERIALS.update(MATERIALS)
    tp.FIXED_KINDS.update((CONVEYOR, TRAY, REJECT))
    for kind in (ROLLER, WORN):
        tp.CUSTOM_KINDS[kind] = (write_roller_usd(kind), roller_object, roller_state, roller_update)


def roller_part(name: str, kind: str, pos, yaw: float = 0.0) -> tp.Part:
    _KIND_OF[name] = kind
    lo, hi = tp.bounds(kind)
    size = tuple(b - a for a, b in zip(lo, hi, strict=True))
    return tp.Part(name=name, pos=tuple(float(v) for v in pos), size=size, mass=None, color=LOOKS["galv"][0],
                   kind=kind, yaw=yaw)
