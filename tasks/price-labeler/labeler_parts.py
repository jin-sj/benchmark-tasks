"""An open cardboard case (a tray) of six soup cans, a hand price labeler in its stand, and the labeler's labels.

The tray is fixed to the table; the cans stand loose in it: each can 66 mm across, 100 mm tall and 0.42 kg, its lid's
centre panel (inside the seam ring) 54 mm across.

The labeler (labeler frame, as it is held: origin between its handle and its trigger, where a hand's grip closes;
x across the grip, from the handle (-x) to the trigger (+x); y along the handle, toward its head (+y); z up): a
handle, a sprung trigger in front of it that the grip squeezes toward the handle, and a head at the handle's +y end
reaching down to a nose roller. One full squeeze of the trigger (``FIRE_MM`` in) and its release feeds one label out
under the nose roller; pressing the roller onto a surface with a label out sticks the label there, under the roller,
along the handle's direction. Squeezed again with a label already out, the labeler feeds another and spoils the
first (it is discarded, beside the stand).

Labels: 25 x 15 mm, white with an orange price band, kept out of sight until fed (the labeler's roll).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import tb_parts as tp
from yam_arena import task_file

MM = 1e-3
CASE = "can_case"
CAN = "soup_can"
LABELER = "price_labeler"
STAND = "labeler_stand"
LABEL = "price_label"
SHELF = "label_shelf"

# -- the case and its cans (case frame: origin on the table under the case's middle) ------------------------------
CAN_R = 33 * MM
CAN_H = 100 * MM
CAN_MASS = 0.42
PANEL_R = 27 * MM  # the lid's centre panel, inside the seam ring
PITCH = 70 * MM  # can to can
ROWS, COLS = 2, 3  # along x, along y
CASE_BASE = 3 * MM
CASE_WALL = (40 * MM, 3 * MM)  # height, thickness
CASE_HALF = (ROWS * PITCH / 2 + 4 * MM, COLS * PITCH / 2 + 4 * MM)


def can_centres() -> list[tuple[float, float]]:
    """Each can's centre (x, y) in the case frame, row by row."""
    return [((r - (ROWS - 1) / 2) * PITCH, (c - (COLS - 1) / 2) * PITCH) for r in range(ROWS) for c in range(COLS)]


LID_Z = CASE_BASE + CAN_H  # the lids' tops above the table

# -- the labeler (labeler frame) ------------------------------------------------------------------------------------
HANDLE = (-20 * MM, -5 * MM, -45 * MM, 45 * MM, -15 * MM, 15 * MM)  # x0, x1, y0, y1, z0, z1
TRIGGER = (5 * MM, 12 * MM, -35 * MM, 35 * MM, -12 * MM, 12 * MM)  # at rest
TRIGGER_TRAVEL = 10 * MM  # toward the handle (-x)
FIRE_MM = 8.0  # squeezed this far in, it feeds a label on release ...
REARM_MM = 2.0  # ... once back out to here
TRIGGER_SPRING = 1500.0  # N/m, preloaded 5 mm: about 7.5 N at rest, 22.5 N squeezed fully in
TRIGGER_PRELOAD = 5 * MM
HEAD = (-20 * MM, 12 * MM, 45 * MM, 85 * MM, -70 * MM, 20 * MM)
NOSE = np.array([-4 * MM, 65 * MM, -78 * MM])  # the roller's lowest point
LABELER_MASS = 0.2
BALANCE = (-4 * MM, 15 * MM, -5 * MM)  # its centre of mass (the label roll in the head balances the handle)
STICK_GAP = 2.0 * MM  # the roller this close to a surface (pressed on it) sticks the label there
UPRIGHT_DEG = 35.0  # ... with the labeler's nose pointing down, within this of vertical

# -- the stand (stand frame: origin on the table under the head's slot) ---------------------------------------------
STAND_BASE = (110 * MM, 70 * MM, 5 * MM)  # along y, along x, thickness
SLOT = (34 * MM, 42 * MM, 62 * MM)  # the head's slot (1 mm of play round the head): across x, along y, wall height


def labeler_in_stand(stand_origin) -> tuple[float, float, float]:
    """Where the labeler's origin sits when it stands in its stand (its nose on the stand's base)."""
    x, y, z = stand_origin
    head_y = (HEAD[2] + HEAD[3]) / 2
    return (float(x - (HEAD[0] + HEAD[1]) / 2), float(y - head_y), float(z + STAND_BASE[2] - NOSE[2] + 0.5 * MM))


# -- labels -----------------------------------------------------------------------------------------------------------
LABEL_SIZE = (25 * MM, 15 * MM, 0.5 * MM)  # along the handle, across, thick
N_LABELS = 20
LABEL_MASS = 0.002

LOOKS = {
    "tin": ((0.78, 0.79, 0.80), 0.3, 0.9),
    "can_wrap": ((0.70, 0.12, 0.10), 0.5, 0.0),  # the cans' printed wrap
    "lid": ((0.82, 0.83, 0.84), 0.25, 0.9),
    "seam": ((0.55, 0.56, 0.58), 0.3, 0.9),
    "cardboard": ((0.76, 0.62, 0.42), 0.8, 0.0),
    "labeler": ((0.10, 0.35, 0.80), 0.5, 0.0),
    "trigger": ((0.12, 0.12, 0.13), 0.6, 0.0),
    "roller": ((0.15, 0.15, 0.15), 0.8, 0.0),
    "label": ((0.97, 0.97, 0.95), 0.7, 0.0),
    "label_band": ((0.95, 0.50, 0.08), 0.7, 0.0),
    "stand": ((0.25, 0.26, 0.28), 0.6, 0.0),
}
MATERIALS = {"cardboard": (0.5, 0.5, "average"), "label": (0.8, 0.8, "max")}


# -- shapes -----------------------------------------------------------------------------------------------------------
def case_shapes() -> list[tp.Shape]:
    """The tray: its floor and four low walls."""
    hx, hy = CASE_HALF
    w, t = CASE_WALL
    b = lambda *v: tp._box(CASE, "cardboard", "cardboard", *v)  # noqa: E731
    return [b(-hx, hx, -hy, hy, 0.0, CASE_BASE),
            b(-hx, -hx + t, -hy, hy, 0.0, w), b(hx - t, hx, -hy, hy, 0.0, w),
            b(-hx, hx, -hy, -hy + t, 0.0, w), b(-hx, hx, hy - t, hy, 0.0, w)]


def can_shapes() -> list[tp.Shape]:
    """A can (can frame: origin at the middle of its bottom, z up its axis)."""
    s = CAN_R * math.cos(math.radians(45))
    z = CAN_H / 2
    return [
        # collider: an eight-pointed section (a square and a cross) inside the can's round
        tp.Shape(CAN, "tin", "steel", (0.0, 0.0, z), (2 * s, 2 * s, CAN_H), visible=False),
        tp.Shape(CAN, "tin", "steel", (0.0, 0.0, z), (2 * CAN_R - 2 * MM, 0.8 * CAN_R, CAN_H), visible=False),
        tp.Shape(CAN, "tin", "steel", (0.0, 0.0, z), (0.8 * CAN_R, 2 * CAN_R - 2 * MM, CAN_H), visible=False),
        # looks: its printed wrap, its tin ends, the lid's seam ring and centre panel
        tp.Shape(CAN, "can_wrap", None, (0.0, 0.0, z), radius=CAN_R, length=CAN_H - 16 * MM),
        tp.Shape(CAN, "tin", None, (0.0, 0.0, 4 * MM), radius=CAN_R, length=8 * MM),
        tp.Shape(CAN, "tin", None, (0.0, 0.0, CAN_H - 4 * MM), radius=CAN_R, length=8 * MM),
        tp.Shape(CAN, "seam", None, (0.0, 0.0, CAN_H + 0.1 * MM), radius=CAN_R, length=0.2 * MM),
        tp.Shape(CAN, "lid", None, (0.0, 0.0, CAN_H + 0.25 * MM), radius=PANEL_R, length=0.2 * MM),
    ]


def stand_shapes() -> list[tp.Shape]:
    ly, lx, t = STAND_BASE
    sx, sy, h = SLOT
    w = 6 * MM
    b = lambda *v: tp._box(STAND, "stand", "plastic", *v)  # noqa: E731
    return [b(-lx / 2, lx / 2, -ly / 2, ly / 2, 0.0, t),
            b(-sx / 2 - w, -sx / 2, -sy / 2 - w, sy / 2 + w, t, h), b(sx / 2, sx / 2 + w, -sy / 2 - w, sy / 2 + w, t, h),
            b(-sx / 2, sx / 2, -sy / 2 - w, -sy / 2, t, h), b(-sx / 2, sx / 2, sy / 2, sy / 2 + w, t, h)]


def labeler_shapes() -> list[tp.Shape]:
    b = lambda body, look, mat, v: tp._box(body, look, mat, *v)  # noqa: E731
    hx0, hx1, hy0, hy1, hz0, hz1 = HEAD
    return [
        b("body", "labeler", "rubber", HANDLE),
        b("body", "labeler", "plastic", HEAD),
        # the nose: the roller's housing and the roller (its lowest point is NOSE)
        tp._box("body", "roller", "rubber", NOSE[0] - 9 * MM, NOSE[0] + 9 * MM, NOSE[1] - 14 * MM, NOSE[1] + 14 * MM,
                NOSE[2], hz0),
        tp.Shape("body", "roller", None, (NOSE[0], NOSE[1], NOSE[2] + 6 * MM), radius=6 * MM, length=20 * MM,
                 axis="Y"),
        # the label roll's window on the head's side
        tp._box("body", "label", None, hx0 - 0.3 * MM, hx0, hy0 + 8 * MM, hy1 - 8 * MM, hz0 + 25 * MM, hz1 - 10 * MM),
        b("trigger", "trigger", "rubber", TRIGGER),
    ]


def label_shapes() -> list[tp.Shape]:
    lx, ly, lz = LABEL_SIZE
    return [tp._box(LABEL, "label", "label", -lx / 2, lx / 2, -ly / 2, ly / 2, 0.0, lz),
            tp._box(LABEL, "label_band", None, -lx / 2 + 2 * MM, lx / 2 - 2 * MM, -ly / 2 + 2 * MM, -ly / 2 + 6 * MM,
                    lz, lz + 0.1 * MM)]


def shelf_shapes() -> list[tp.Shape]:
    """Where the labels wait, out of sight under the table: the labeler's roll."""
    return [tp._box(SHELF, "stand", "plastic", -0.02, 0.02, -0.15, 0.15, 0.0, 0.005)]


# -- the labeler: an articulation (its body and its sprung trigger) ------------------------------------------------
def write_labeler_usd(path) -> None:
    from pxr import Gf, Sdf, UsdPhysics

    stage, root, looks, mats = tp._new_stage(path)
    paths = {}
    for body, mass in (("body", LABELER_MASS - 0.02), ("trigger", 0.02)):
        p = f"/Part/{body}"
        prim = tp._xform(stage, p)
        paths[body] = p
        UsdPhysics.RigidBodyAPI.Apply(prim)
        m = UsdPhysics.MassAPI.Apply(prim)
        m.CreateMassAttr(mass)
        if body == "body":
            m.CreateCenterOfMassAttr(Gf.Vec3f(*BALANCE))
        tp._apply(prim, "PhysxRigidBodyAPI", {
            "physxRigidBody:solverPositionIterationCount": 32,
            "physxRigidBody:solverVelocityIterationCount": 4,
            "physxRigidBody:maxDepenetrationVelocity": 0.3,
        })
    tp._write_shapes(stage, LABELER, paths, looks, mats)
    root_prim = stage.GetPrimAtPath(paths["body"])
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    tp._apply(root_prim, "PhysxArticulationAPI", {
        "physxArticulation:enabledSelfCollisions": False,
        "physxArticulation:solverPositionIterationCount": 32,
        "physxArticulation:solverVelocityIterationCount": 4,
    })
    j = UsdPhysics.PrismaticJoint.Define(stage, "/Part/joints/trigger")
    j.CreateBody0Rel().SetTargets([Sdf.Path(paths["body"])])
    j.CreateBody1Rel().SetTargets([Sdf.Path(paths["trigger"])])
    j.CreateAxisAttr("X")  # negative: squeezed in, toward the handle
    j.CreateLowerLimitAttr(-TRIGGER_TRAVEL)
    j.CreateUpperLimitAttr(0.0)
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(TRIGGER_SPRING)
    drive.CreateDampingAttr(3.0)
    drive.CreateTargetPositionAttr(TRIGGER_PRELOAD)
    tp._apply(j.GetPrim(), "PhysxJointAPI", {"physxJoint:armature": 1e-3})
    stage.GetRootLayer().Save()


def labeler_object(spec: tp.Part):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    from isaaclab_physx.assets.articulation import Articulation as PhysxArticulation

    class _LabelerArticulation(PhysxArticulation):
        def _spring(self):
            import torch

            self.set_joint_position_target(torch.full((self.num_instances, self.num_joints), TRIGGER_PRELOAD,
                                                      device=self.device))

        def reset(self, env_ids=None, env_mask=None):
            import torch

            super().reset(env_ids, env_mask)
            q = torch.zeros((self.num_instances, self.num_joints), device=self.device)
            self.write_joint_position_to_sim_index(position=q)
            self.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
            self._spring()

        def set_joint_position_target_index(self, *, target, joint_ids=None, env_ids=None, full_data=False):
            super().set_joint_position_target_index(target=target, joint_ids=joint_ids, env_ids=env_ids,
                                                    full_data=full_data)
            self._spring()

    class _Labeler(Object):
        def _generate_articulation_cfg(self):
            cfg = super()._generate_articulation_cfg()
            cfg.class_type = _LabelerArticulation
            cfg.actuators = {"trigger": ImplicitActuatorCfg(joint_names_expr=["trigger"], stiffness=TRIGGER_SPRING,
                                                            damping=3.0, armature=1e-3)}
            cfg.init_state.joint_pos = {"trigger": 0.0}
            return cfg

    cfg = sim_utils.UsdFileCfg(usd_path=tp.usd_file(LABELER), fix_root_link=False)
    pose = Pose(position_xyz=tuple(spec.pos), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    return _Labeler(name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=cfg, initial_pose=pose)


@dataclass(frozen=True)
class LabelerState(task_file.ObjectState):
    """The labeler's ``ObjectState`` (pos: its bounds' centre) plus its origin (where the grip closes) and how far its
    trigger is squeezed in (mm)."""

    origin: tuple = (0.0, 0.0, 0.0)
    trigger_mm: float = 0.0


def labeler_state(base, name: str) -> LabelerState:
    art = base.scene.articulations[name]
    origin = base.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
    pose = art.data.root_pose_w.torch[0].detach().cpu().numpy().astype(np.float64)
    q = art.data.joint_pos.torch[0].detach().cpu().numpy().astype(np.float64)
    from yam_arena.urdf import quat_xyzw_to_matrix

    o = pose[:3] - origin
    rot = quat_xyzw_to_matrix(pose[3:])
    lo, hi = tp.bounds(LABELER)
    return LabelerState(pos=o + rot @ np.array(tp.local_center(LABELER)), quat_xyzw=pose[3:], lin_vel=np.zeros(3),
                        ang_vel=np.zeros(3), low=o + np.array(lo), high=o + np.array(hi), origin=tuple(float(v) for v in o),
                        trigger_mm=-float(q[0]) * 1000.0)


def install() -> None:
    tp.SHAPES.update({CASE: case_shapes, CAN: can_shapes, STAND: stand_shapes, LABELER: labeler_shapes,
                      LABEL: label_shapes, SHELF: shelf_shapes})
    tp.LOOKS.update(LOOKS)
    tp.MATERIALS.update(MATERIALS)
    tp.FIXED_KINDS.update((CASE, STAND, SHELF))
    tp.FREE_KINDS[LABEL] = LABEL_MASS
    tp.FREE_KINDS[CAN] = CAN_MASS
    tp.CUSTOM_KINDS[LABELER] = (write_labeler_usd, labeler_object, labeler_state)


def labeler_part(name: str, pos) -> tp.Part:
    lo, hi = tp.bounds(LABELER)
    size = tuple(b - a for a, b in zip(lo, hi, strict=True))
    return tp.Part(name=name, pos=tuple(float(v) for v in pos), size=size, mass=None, color=LOOKS["labeler"][0],
                   kind=LABELER)
