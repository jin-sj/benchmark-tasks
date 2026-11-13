"""A wall-mounted split air conditioner's indoor unit (on a wall board) with a hinged front cover, and two mesh air
filters (a dusty one fitted in the unit, a clean one).

Unit frame: origin on the table under the middle of the unit's front face; +x into the unit (toward the wall, away
from the arms), y along it (+y the robot's left), z up. The unit hangs on the wall board, its underside
``BOTTOM`` above the table.

The cover is one hinged body: a panel over the unit's front, hinged along its top edge. Its angle is 0 closed
(hanging down over the front) and grows as its bottom edge swings out and up, to a stop at ``OPEN_DEG``. Its hinge
is a friction hinge: the cover stays at whatever angle it is left (the hinge's drive is re-aimed every control step
at the cover's angle, plus what holds it against its weight).

Filter frame: origin at the middle of the filter's bottom edge, on its front face; +x through it toward the unit,
y along it, z up its face. Its lower part is flat and upright; its top part bends back (toward +x) by ``BEND_DEG``,
following the unit's coil. Two pull tabs stick out of its front at the bottom.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import tb_parts as tp
from yam_arena import task_file

MM = 1e-3
UNIT = "ac_unit"
FILTER_DUSTY = "filter_dusty"
FILTER_CLEAN = "filter_clean"

# -- the filter -----------------------------------------------------------------------------------------------------
WIDTH = 600 * MM  # along y
LOW_H = 115 * MM  # its flat, upright lower part
TOP_H = 55 * MM  # its top part, bent back
BEND_DEG = 15.0
THICK = 5 * MM  # the frame's depth
BORDER = 8 * MM  # the frame's border width
TAB = (20 * MM, 20 * MM, 3 * MM)  # each pull tab: how far it sticks out, its width, its thickness
TAB_Y = 150 * MM  # the tabs, either side of the middle
FILTER_MASS = {FILTER_DUSTY: 0.09, FILTER_CLEAN: 0.08}
MESH = {FILTER_DUSTY: "mesh_dusty", FILTER_CLEAN: "mesh_clean"}


def bend() -> tuple[float, float]:
    """(how far back, how far up) the top part's top edge is from its bottom edge."""
    b = math.radians(BEND_DEG)
    return TOP_H * math.sin(b), TOP_H * math.cos(b)


FILTER_H = LOW_H + bend()[1]  # its overall height

# -- the unit (unit frame) -------------------------------------------------------------------------------------------
BOTTOM = 80 * MM  # the unit's underside above the table
LEDGE = 150 * MM  # the filter stands on this ledge (the top of the unit's outlet section)
ROOF = 335 * MM  # the filter's cavity's roof: the unit's top cap's underside
TOP = 350 * MM  # the unit's top
DEPTH = 190 * MM
HALF_W = 350 * MM
SEAT_X = 6 * MM  # a fitted filter's front face (its lower part), behind the front's plane
GAP = 5 * MM  # between a fitted filter's back and the coil (lifted by its tabs, in front of it, its top tips back a little)
COIL_TOP_BACK = 6 * MM  # the coil's upper part (behind the filter's bent top) is set this much further back
LUG = (3 * MM, 6 * MM, 30 * MM)  # the lugs in front of the filter's bottom edge: depth, height, width
LUG_Y = (-220 * MM, 0.0, 220 * MM)
LIP_Z = 300 * MM  # the lip in front of the filter's top, from here up to the roof ...
LIP_BACK = 11 * MM  # ... its back face this far behind the front's plane
WALL = (20 * MM, 900 * MM, 450 * MM)  # the wall board behind the unit: thickness, width, height

# -- the cover ------------------------------------------------------------------------------------------------------
HINGE = (-6 * MM, 344 * MM)  # the hinge axis (x, z), along y
PANEL = (165 * MM, 338 * MM, 12 * MM)  # the panel (closed): bottom and top edges' heights, thickness
PULL = (10 * MM, 5 * MM)  # the finger pull along its bottom edge: how far it sticks out, its height
COVER_MASS = 0.3
COVER_COM = 92.5 * MM  # its centre of mass from the hinge (the panel's middle)
OPEN_DEG = 80.0  # its upper stop
HOLD = 10.0  # N m/rad: the friction hinge's stiffness about where it was left ...
SLIP_DEG = 1.0  # ... until pushed this far from there: then where it is held moves with it (about 0.17 N m, 1 N at
# the cover's bottom edge, moves it)
DAMPING = 0.05
ARMATURE = 1e-4

LOOKS = {
    "unit": ((0.93, 0.93, 0.91), 0.35, 0.0),  # white plastic
    "unit_dark": ((0.20, 0.21, 0.22), 0.6, 0.0),  # the outlet, the coil behind the filter
    "coil": ((0.62, 0.64, 0.66), 0.4, 0.6),
    "wall": ((0.86, 0.83, 0.76), 0.9, 0.0),
    "led": ((0.15, 0.55, 0.95), 0.3, 0.0),
    "frame": ((0.55, 0.58, 0.62), 0.6, 0.0),  # the filter's grey plastic frame
    "mesh_clean": ((0.80, 0.82, 0.84), 0.8, 0.0),
    "mesh_dusty": ((0.47, 0.44, 0.39), 0.95, 0.0),
}


# -- shapes ---------------------------------------------------------------------------------------------------------
def _filter_shapes(kind: str) -> list[tp.Shape]:
    hw, t, b = WIDTH / 2, THICK, BORDER
    bx, bz = bend()
    mesh = MESH[kind]
    box = lambda *v, look="frame", mat="plastic": tp._box(kind, look, mat, *v)  # noqa: E731

    def slant(y0, y1, look, mat=None, z0=0.0, z1=None, visible=True, x0=0.0, x1=t):
        """A slab of the bent top part between heights (along it) z0..z1 of its TOP_H, depth x0..x1."""
        z1 = TOP_H if z1 is None else z1
        s, c = math.sin(math.radians(BEND_DEG)), math.cos(math.radians(BEND_DEG))
        p = [(x0 + z0 * s, LOW_H + z0 * c), (x1 + z0 * s, LOW_H + z0 * c), (x1 + z1 * s, LOW_H + z1 * c),
             (x0 + z1 * s, LOW_H + z1 * c)]
        return tp._prism(kind, look, mat, p, y0, y1, visible=visible)

    out = [
        # collision: the flat lower part, the bent top part, the tabs
        tp.Shape(kind, mesh, "plastic", (t / 2, 0.0, LOW_H / 2), (t, WIDTH, LOW_H), visible=False),
        slant(-hw, hw, mesh, "plastic", visible=False),
    ]
    for s in (-1, 1):
        out.append(box(-TAB[0], 0.0, s * TAB_Y - TAB[1] / 2, s * TAB_Y + TAB[1] / 2, 0.0, TAB[2]))
    # looks: the frame (border and ribs) a little proud of the mesh, front and back
    d = 0.3 * MM
    for x0, x1 in ((-d, 0.0), (t, t + d)):
        out += [box(x0, x1, -hw, hw, 0.0, b, mat=None), box(x0, x1, -hw, -hw + b, b, LOW_H, mat=None),
                box(x0, x1, hw - b, hw, b, LOW_H, mat=None), box(x0, x1, -hw + b, hw - b, b, LOW_H, look=mesh, mat=None)]
        for y in np.arange(-hw + 75 * MM, hw - 50 * MM, 75 * MM):
            out.append(box(x0, x1, y - 1.5 * MM, y + 1.5 * MM, b, LOW_H, mat=None))
        out.append(box(x0, x1, -hw, hw, LOW_H / 2 - 1.5 * MM, LOW_H / 2 + 1.5 * MM, mat=None))
        out += [slant(-hw, -hw + b, "frame", x0=x0, x1=x1), slant(hw - b, hw, "frame", x0=x0, x1=x1),
                slant(-hw + b, hw - b, mesh, x0=x0, x1=x1, z1=TOP_H - b),
                slant(-hw, hw, "frame", x0=x0, x1=x1, z0=TOP_H - b)]
    return out


def dusty_shapes():
    return _filter_shapes(FILTER_DUSTY)


def clean_shapes():
    return _filter_shapes(FILTER_CLEAN)


def unit_shapes() -> list[tp.Shape]:
    hw, d = HALF_W, DEPTH
    b = lambda *v, look="unit", mat="plastic": tp._box("base", look, mat, *v)  # noqa: E731
    bx, bz = bend()
    s = math.tan(math.radians(BEND_DEG))
    coil_x = SEAT_X + THICK + GAP  # the coil's face behind the filter's lower part
    out = [
        # the wall board behind the unit
        b(d, d + WALL[0], -WALL[1] / 2, WALL[1] / 2, 0.0, WALL[2], look="wall"),
        # the outlet section below the filter: its top is the filter's ledge
        b(0.0, d, -hw, hw, BOTTOM, LEDGE),
        b(-0.5 * MM, 0.0, -hw + 40 * MM, hw - 40 * MM, BOTTOM + 15 * MM, BOTTOM + 45 * MM, look="unit_dark", mat=None),
        # the top cap above the filter's cavity, and the ends
        b(0.0, d, -hw, hw, ROOF, TOP),
        b(0.0, d, -hw, -WIDTH / 2 - 15 * MM, LEDGE, ROOF), b(0.0, d, WIDTH / 2 + 15 * MM, hw, LEDGE, ROOF),
        # the coil behind the filter: upright behind its lower part, leaning back behind its top part
        b(coil_x, d, -WIDTH / 2 - 15 * MM, WIDTH / 2 + 15 * MM, LEDGE, LEDGE + LOW_H, look="coil", mat="steel"),
        tp._prism("base", "coil", "steel", ((coil_x + COIL_TOP_BACK, LEDGE + LOW_H), (d, LEDGE + LOW_H), (d, ROOF),
                                             (coil_x + COIL_TOP_BACK + (ROOF - LEDGE - LOW_H) * s, ROOF)),
                  -WIDTH / 2 - 15 * MM, WIDTH / 2 + 15 * MM),
        # the lip in front of the filter's top
        b(0.0, LIP_BACK, -WIDTH / 2 - 15 * MM, WIDTH / 2 + 15 * MM, LIP_Z, ROOF),
    ]
    for y in LUG_Y:  # the lugs in front of the filter's bottom edge
        out.append(b(0.0, LUG[0], y - LUG[2] / 2, y + LUG[2] / 2, LEDGE, LEDGE + LUG[1]))
    # the cover (closed): the panel and the finger pull along its bottom edge; a status light
    z0, z1, pt = PANEL
    c = lambda *v, look="unit", mat="plastic": tp._box("cover", look, mat, *v)  # noqa: E731
    out += [c(-pt, 0.0, -hw, hw, z0, z1), c(-pt - PULL[0], -pt, -hw + 60 * MM, hw - 60 * MM, z0, z0 + PULL[1]),
            c(-pt - 0.3 * MM, -pt, hw - 50 * MM, hw - 40 * MM, z0 + 20 * MM, z0 + 26 * MM, look="led", mat=None)]
    return out


# -- the unit's USD and Arena object (an articulation: the fixed body and the hinged cover) ------------------------
def write_usd(path) -> None:
    from pxr import Gf, Sdf, UsdPhysics

    stage, root, looks, mats = tp._new_stage(path)
    paths = {}
    for body in ("base", "cover"):
        p = f"/Part/{body}"
        prim = tp._xform(stage, p)
        paths[body] = p
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(8.0 if body == "base" else COVER_MASS)
        tp._apply(prim, "PhysxRigidBodyAPI", {
            "physxRigidBody:solverPositionIterationCount": 32,
            "physxRigidBody:solverVelocityIterationCount": 4,
            "physxRigidBody:maxDepenetrationVelocity": 0.3,
        })
    tp._write_shapes(stage, UNIT, paths, looks, mats)
    base_prim = stage.GetPrimAtPath(paths["base"])
    UsdPhysics.ArticulationRootAPI.Apply(base_prim)
    tp._apply(base_prim, "PhysxArticulationAPI", {
        "physxArticulation:enabledSelfCollisions": False,
        "physxArticulation:solverPositionIterationCount": 32,
        "physxArticulation:solverVelocityIterationCount": 4,
    })
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Part/joints/cover")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(paths["base"])])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(paths["cover"])])
    joint.CreateAxisAttr("Y")  # about +y a positive angle swings the cover's bottom edge out (-x) and up
    joint.CreateLowerLimitAttr(0.0)
    joint.CreateUpperLimitAttr(OPEN_DEG)
    hinge = Gf.Vec3f(HINGE[0], 0.0, HINGE[1])
    joint.CreateLocalPos0Attr(hinge)
    joint.CreateLocalPos1Attr(hinge)
    joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(math.radians(HOLD))
    drive.CreateDampingAttr(math.radians(DAMPING))
    drive.CreateTargetPositionAttr(0.0)
    tp._apply(joint.GetPrim(), "PhysxJointAPI", {"physxJoint:armature": ARMATURE})
    stage.GetRootLayer().Save()


def _cover_index(art) -> int:
    return list(art.joint_names).index("cover")


_HELD: dict = {}  # per articulation: the angle (rad) the friction hinge holds the cover at


def _set_spring(art, deg: float, reset: bool = False) -> None:
    """The friction hinge: hold the cover where it was left, against its weight; pushed more than SLIP_DEG from
    there, it slips (where it is held follows it)."""
    import torch

    q = math.radians(deg)
    held = q if reset or id(art) not in _HELD else _HELD[id(art)]
    slip = math.radians(SLIP_DEG)
    held = min(max(held, q - slip), q + slip)
    _HELD[id(art)] = held
    weight = COVER_MASS * 9.81 * COVER_COM * math.sin(held)  # its weight's torque, closing it
    target = held + weight / HOLD
    art.set_joint_position_target(torch.full((art.num_instances, art.num_joints), target, device=art.device))


def arena_object(spec: tp.Part):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    from isaaclab_physx.assets.articulation import Articulation as PhysxArticulation

    class _UnitArticulation(PhysxArticulation):
        def reset(self, env_ids=None, env_mask=None):
            # the cover shut, at rest
            import torch

            super().reset(env_ids, env_mask)
            q = torch.zeros((self.num_instances, self.num_joints), device=self.device)
            self.write_joint_position_to_sim_index(position=q)
            self.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
            _set_spring(self, 0.0, reset=True)

        def set_joint_position_target_index(self, *, target, joint_ids=None, env_ids=None, full_data=False):
            # only reset events write targets this way (they would set the target to the default)
            super().set_joint_position_target_index(target=target, joint_ids=joint_ids, env_ids=env_ids,
                                                    full_data=full_data)
            _set_spring(self, 0.0, reset=True)

    class _Unit(Object):
        def _generate_articulation_cfg(self):
            cfg = super()._generate_articulation_cfg()
            cfg.class_type = _UnitArticulation
            cfg.actuators = {"cover": ImplicitActuatorCfg(joint_names_expr=["cover"], stiffness=HOLD,
                                                          damping=DAMPING, armature=ARMATURE)}
            cfg.init_state.joint_pos = {"cover": 0.0}
            return cfg

    cfg = sim_utils.UsdFileCfg(usd_path=tp.usd_file(UNIT), fix_root_link=True)
    pose = Pose(position_xyz=tuple(spec.pos), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    return _Unit(name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=cfg, initial_pose=pose)


@dataclass(frozen=True)
class UnitState(task_file.ObjectState):
    """The unit's ``ObjectState`` (its origin) plus its cover's angle in degrees (0 shut, OPEN_DEG at the stop)."""

    cover_deg: float = 0.0


def state(base, name: str) -> UnitState:
    art = base.scene.articulations[name]
    origin = base.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
    pose = art.data.root_pose_w.torch[0].detach().cpu().numpy().astype(np.float64)
    pos = pose[:3] - origin
    q = art.data.joint_pos.torch[0].detach().cpu().numpy().astype(np.float64)
    lo, hi = tp.bounds(UNIT)
    return UnitState(pos=pos + np.array(tp.local_center(UNIT)), quat_xyzw=pose[3:], lin_vel=np.zeros(3),
                     ang_vel=np.zeros(3), low=pos + np.array(lo), high=pos + np.array(hi),
                     cover_deg=float(math.degrees(q[_cover_index(art)])))


def update(base, name: str, out) -> None:
    """Once per control step: the friction hinge holds the cover where it is."""
    art = base.scene.articulations[name]
    deg = math.degrees(float(art.data.joint_pos.torch[0, _cover_index(art)]))
    _set_spring(art, deg)


def install() -> None:
    tp.SHAPES[UNIT] = unit_shapes
    tp.SHAPES[FILTER_DUSTY] = dusty_shapes
    tp.SHAPES[FILTER_CLEAN] = clean_shapes
    tp.LOOKS.update(LOOKS)
    tp.CUSTOM_KINDS[UNIT] = (write_usd, arena_object, state, update)
    tp.FREE_KINDS.update(FILTER_MASS)


def unit_part(name: str, pos) -> tp.Part:
    lo, hi = tp.bounds(UNIT)
    size = tuple(b - a for a, b in zip(lo, hi, strict=True))
    return tp.Part(name=name, pos=tuple(float(v) for v in pos), size=size, mass=None, color=LOOKS["unit"][0],
                   kind=UNIT)


# -- poses (world) ---------------------------------------------------------------------------------------------------
def frame(state) -> tuple[np.ndarray, np.ndarray]:
    """(origin, rotation) of a filter's frame, from its ``ObjectState`` (whose pos is its bounds' centre)."""
    from yam_arena.urdf import quat_xyzw_to_matrix

    rot = quat_xyzw_to_matrix(np.asarray(state.quat_xyzw, dtype=float))
    return np.asarray(state.pos, dtype=float) - rot @ np.asarray(tp.local_center(FILTER_CLEAN), dtype=float), rot


def points(state) -> np.ndarray:
    """A filter's corners (its frame's outline, front and back, and its top edge), in the world."""
    o, rot = frame(state)
    bx, bz = bend()
    hw = WIDTH / 2
    local = [(x, y, z) for y in (-hw, hw) for x, z in ((0.0, 0.0), (THICK, 0.0), (0.0, LOW_H), (THICK, LOW_H),
                                                         (bx, FILTER_H), (bx + THICK, FILTER_H))]
    return np.array([o + rot @ np.array(p) for p in local])
