"""A handheld digital multimeter of this task's own generic design: an autoranging 6000-count meter in a rubber
holster, lying face up on the table, with a five-position rotary selector and a seven-segment LCD that shows what
the meter reads.

One articulation with a fixed root (the meter's case, held to the table by a hook-and-loop strip) and one hinged
link, the selector knob, on a revolute joint about the face's normal. The knob clicks between five positions 30
degrees apart (clockwise, seen from above): OFF, V~ (AC volts), V DC (DC volts, its pointer straight at the
display), mV (DC millivolts) and Ohm, with stops at OFF and Ohm. The click is a spring pulled toward the nearest
position (re-aimed every control step): about 0.12 N m to turn it from one position to the next, as on a real
meter's selector.

The display is a row of seven-segment characters (a minus sign and four digits with decimal points), 22 mm tall,
drawn as thin dark strokes over the pale LCD window; at every reading :func:`show` raises each stroke to show
over the LCD and moves every other one under its surface, so the cameras see the reading. The meter's reading itself is worked out in ``probe_parts``.

Frame: the meter's origin is the middle of its underside; x along the meter toward its display end (away from the
arms), y to the robot's left, z up. Seen by a person at the arms' side, the display reads left to right along -y.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import tb_parts as tp
from yam_arena import task_file

MM = 1e-3
KIND = "multimeter"

# -- geometry (meter frame, metres) ---------------------------------------------------------------------------
CASE = (170 * MM, 90 * MM, 42 * MM)  # holster: length (x), width (y), height of its rim
FACE_TOP = 44 * MM  # the face (black panel) inside the holster's rim
DIAL = (-30 * MM, 0.0)  # the selector's axis (x, y) on the face
KNOB_R = 20 * MM  # the round knob, 5 mm tall on the face
KNOB_TOP = FACE_TOP + 5 * MM
BAR = (38 * MM, 8 * MM, 9 * MM)  # the grip bar across the knob: length (along the pointer), width, height above it
LCD = (32 * MM, 78 * MM, -37 * MM, 37 * MM)  # the display window: x0, x1, y0, y1
DIGIT_H, DIGIT_W, STROKE = 22 * MM, 11 * MM, 3.0 * MM
DIGIT_PITCH = 16 * MM  # a 5 mm gap between digits (the decimal point sits in it)
MINUS_W = 6 * MM
LCD_MID_X = (LCD[0] + LCD[1]) / 2
METER_MASS = 0.45  # kg with its holster (fixed to the table in this task)
KNOB_MASS = 0.01

# -- the selector ----------------------------------------------------------------------------------------------
# joint angle (deg, counter-clockwise seen from above; the pointer at 0 points at the display): OFF at +60, then
# clockwise in 30 degree steps
POSITIONS = {"OFF": 60.0, "V~": 30.0, "V DC": 0.0, "mV": -30.0, "Ohm": -60.0}
STEP_DEG = 30.0
CLICK_TORQUE = 0.12  # N m at the half-way point between two positions
SPRING = CLICK_TORQUE / math.radians(STEP_DEG / 2)  # N m / rad
DAMPING = 0.002  # N m s / rad
ARMATURE = 2e-5  # kg m2
IN_POSITION_DEG = 8.0  # within this of a position, the switch makes that position's contacts

LOOKS = {"holster": ((0.22, 0.24, 0.26), 0.7, 0.0), "face": ((0.06, 0.06, 0.07), 0.5, 0.0),
         "lcd": ((0.66, 0.71, 0.62), 0.4, 0.0), "segment": ((0.03, 0.03, 0.04), 0.5, 0.0),
         "knob": ((0.12, 0.12, 0.13), 0.5, 0.0), "marking": ((0.93, 0.93, 0.90), 0.6, 0.0),
         "jack_red": ((0.80, 0.10, 0.08), 0.5, 0.0), "jack_black": ((0.02, 0.02, 0.02), 0.5, 0.0)}


def _shapes() -> list[tp.Shape]:
    """Colliders and solid looks: the holster and face (case), the knob and its grip bar (knob, in the knob's
    frame: origin on the dial's axis at the face, the pointer along +x)."""
    lx, wy, h = CASE
    cx, cy = DIAL
    out = [
        tp._box("case", "holster", "rubber", -lx / 2, lx / 2, -wy / 2, wy / 2, 0.0, h),
        tp._box("case", "face", None, -lx / 2 + 5 * MM, lx / 2 - 5 * MM, -wy / 2 + 5 * MM, wy / 2 - 5 * MM, h, FACE_TOP),
        tp._box("case", "lcd", None, LCD[0], LCD[1], LCD[2], LCD[3], FACE_TOP, FACE_TOP + 0.4 * MM),
        # the input jacks at the meter's near end: COM (black) and V (red), 4 mm sockets in raised collars
        tp.Shape("case", "jack_black", None, center=(-70 * MM, 14 * MM, FACE_TOP + 1 * MM), radius=5 * MM,
                 length=2 * MM, axis="Z"),
        tp.Shape("case", "jack_red", None, center=(-70 * MM, -14 * MM, FACE_TOP + 1 * MM), radius=5 * MM,
                 length=2 * MM, axis="Z"),
        tp.Shape("knob", "knob", None, center=(0.0, 0.0, 2.5 * MM), radius=KNOB_R, length=5 * MM, axis="Z"),
        tp._box("knob", "knob", "plastic", -BAR[0] / 2, BAR[0] / 2, -BAR[1] / 2, BAR[1] / 2, 5 * MM, 5 * MM + BAR[2]),
        tp._box("knob", "marking", None, BAR[0] / 2 - 14 * MM, BAR[0] / 2 - 1.5 * MM, -1.5 * MM, 1.5 * MM,
                5 * MM + BAR[2], 5 * MM + BAR[2] + 0.3 * MM),  # the pointer's white line
    ]
    del cx, cy
    return out


# -- strokes: thin flat quads on the face (labels, ticks, the display's segments) -------------------------------
def _uv_to_xy(u: float, v: float, origin=(0.0, 0.0)) -> tuple[float, float]:
    """Face coordinates as a reader at the arms' side sees them (u to the right, v up) to the meter frame."""
    return origin[0] + v, origin[1] - u


def _segments(u0: float, v0: float) -> dict[str, tuple[float, float, float, float]]:
    """A seven-segment digit's segments and its decimal point, (u0, u1, v0, v1) rectangles, its lower left corner
    at (u0, v0)."""
    W, H, s = DIGIT_W, DIGIT_H, STROKE
    g = 0.25 * MM  # the gap between neighbouring segments
    r = {
        "a": (s / 2 + g, W - s / 2 - g, H - s, H), "d": (s / 2 + g, W - s / 2 - g, 0.0, s),
        "g": (s / 2 + g, W - s / 2 - g, H / 2 - s / 2, H / 2 + s / 2),
        "b": (W - s, W, H / 2 + g, H - s / 2 - g), "c": (W - s, W, s / 2 + g, H / 2 - g),
        "f": (0.0, s, H / 2 + g, H - s / 2 - g), "e": (0.0, s, s / 2 + g, H / 2 - g),
        "p": (W + 0.6 * MM, W + 0.6 * MM + s, 0.0, s),
    }
    return {k: (u0 + a, u0 + b, v0 + c, v0 + d) for k, (a, b, c, d) in r.items()}


N_DIGITS = 4


def display_cells() -> dict[str, tuple[float, float, float, float]]:
    """Every stroke of the display: 'm' (the minus sign) and 'k.x' (digit k from the left, segment x)."""
    total = MINUS_W + 1 * MM + N_DIGITS * DIGIT_PITCH - (DIGIT_PITCH - DIGIT_W)
    u = -total / 2
    v0 = -DIGIT_H / 2
    cells = {"m": (u, u + MINUS_W, v0 + DIGIT_H / 2 - STROKE / 2, v0 + DIGIT_H / 2 + STROKE / 2)}
    u += MINUS_W + 1 * MM
    for k in range(N_DIGITS):
        for seg, rect in _segments(u + k * DIGIT_PITCH, v0).items():
            cells[f"{k}.{seg}"] = rect
    return cells


GLYPHS = {" ": "", "0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd", "6": "afgedc",
          "7": "abc", "8": "abcdefg", "9": "abcdfg", "O": "abcdef", "L": "def", "-": "g"}


def text_cells(text: str) -> set[str]:
    """The strokes to show for ``text``: an optional leading '-', then up to four characters, each optionally
    followed by '.', right-aligned (as on a meter: '12.00', '-12.00', '0.00', 'OL', '')."""
    on: set[str] = set()
    if text.startswith("-"):
        on.add("m")
        text = text[1:]
    chars: list[tuple[str, bool]] = []
    for ch in text:
        if ch == "." and chars:
            chars[-1] = (chars[-1][0], True)
        else:
            chars.append((ch, False))
    chars = [(" ", False)] * (N_DIGITS - len(chars)) + chars
    for k, (ch, dot) in enumerate(chars):
        on |= {f"{k}.{s}" for s in GLYPHS[ch]}
        if dot:
            on.add(f"{k}.p")
    return on


def _stroke_font() -> dict[str, list[tuple[float, float, float, float]]]:
    """A small stroke font for the selector's labels, in units of the letter height: (u0, v0, u1, v1) lines."""
    return {
        "O": [(0, 0, 0.6, 0), (0.6, 0, 0.6, 1), (0.6, 1, 0, 1), (0, 1, 0, 0)],
        "F": [(0, 0, 0, 1), (0, 1, 0.55, 1), (0, 0.5, 0.45, 0.5)],
        "V": [(0, 1, 0.3, 0), (0.3, 0, 0.6, 1)],
        "~": [(0, 0.45, 0.15, 0.6), (0.15, 0.6, 0.3, 0.45), (0.3, 0.45, 0.45, 0.3), (0.45, 0.3, 0.6, 0.45)],
        "=": [(0, 0.65, 0.7, 0.65), (0, 0.35, 0.15, 0.35), (0.27, 0.35, 0.43, 0.35), (0.55, 0.35, 0.7, 0.35)],
        "m": [(0, 0, 0, 0.6), (0, 0.6, 0.5, 0.6), (0.25, 0.6, 0.25, 0), (0.5, 0.6, 0.5, 0)],
        "W": [(0.05, 0, 0.2, 0), (0.2, 0, 0.12, 0.35), (0.12, 0.35, 0.12, 0.7), (0.12, 0.7, 0.3, 1),
              (0.3, 1, 0.48, 0.7), (0.48, 0.7, 0.48, 0.35), (0.48, 0.35, 0.4, 0), (0.4, 0, 0.55, 0)],  # Ohm sign
    }


LABELS = {"OFF": "OFF", "V~": "V~", "V DC": "V=", "mV": "mV", "Ohm": "W"}
LABEL_H = 6 * MM
LABEL_R = 32 * MM  # labels' centres from the dial's axis
TICK_R = (22 * MM, 26 * MM)


def marking_strokes() -> list[tuple[float, float, float, float, float]]:
    """The selector's position ticks and labels on the face, as (x0, y0, x1, y1, width) lines in the meter frame."""
    font = _stroke_font()
    out = []
    for pos, deg in POSITIONS.items():
        a = math.radians(deg)
        d = np.array([math.cos(a), math.sin(a)])
        p0, p1 = np.array(DIAL) + TICK_R[0] * d, np.array(DIAL) + TICK_R[1] * d
        out.append((*p0, *p1, 1.2 * MM))
        text = LABELS[pos]
        width = sum(0.85 for _ in text) * LABEL_H
        cx, cy = np.array(DIAL) + LABEL_R * d + np.array([0.0, 0.0])
        u_c, v_c = -(cy - 0.0), cx  # label centre in the reader's (u, v), then laid out left to right
        u = u_c - width / 2
        for ch in text:
            for a0, b0, a1, b1 in font[ch]:
                x0, y0 = _uv_to_xy(u + a0 * LABEL_H, v_c - LABEL_H / 2 + b0 * LABEL_H)
                x1, y1 = _uv_to_xy(u + a1 * LABEL_H, v_c - LABEL_H / 2 + b1 * LABEL_H)
                out.append((x0, y0, x1, y1, 0.9 * MM))
            u += 0.85 * LABEL_H
    # the jacks' labels: "COM" is not drawn; a short line under each jack
    return out


def install() -> None:
    tp.SHAPES[KIND] = _shapes
    tp.LOOKS.update(LOOKS)
    tp.CUSTOM_KINDS[KIND] = (write_usd, arena_object, state, update)


def part(name: str, pos) -> tp.Part:
    """The meter with its origin at ``pos`` (world, m). Fixed case: no mass for yam_arena."""
    lo, hi = tp.bounds(KIND)
    size = tuple(b - a for a, b in zip(lo, hi, strict=True))
    return tp.Part(name=name, pos=tuple(float(v) for v in pos), size=size, mass=None, color=LOOKS["holster"][0],
                   kind=KIND)


HIDDEN_DROP = 3 * MM  # a hidden segment is moved this far down, under the LCD's surface (inside the case)
SHOWN_LIFT = 4 * MM  # a shown segment is drawn this far above the LCD (no shadow): far enough in depth that the
# renderer's temporal filter treats the LCD showing through again as newly uncovered, not as the old segment


def _quad(stage, path: str, corners, z: float, look, movable: bool = False) -> None:
    from pxr import Gf, UsdGeom, UsdShade, Vt

    geom = UsdGeom.Mesh.Define(stage, path)
    if movable:  # a display segment: shown at z, hidden HIDDEN_DROP lower (starts hidden: the meter is off)
        geom.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -HIDDEN_DROP))
        from pxr import Sdf

        geom.GetPrim().CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
    geom.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(float(x), float(y), z) for x, y in corners]))
    geom.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    geom.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    geom.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    geom.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(look)


def write_usd(path) -> None:
    from pxr import Gf, Sdf, UsdPhysics

    stage, root, looks, mats = tp._new_stage(path)
    paths = {}
    for body, mass in (("case", METER_MASS), ("knob", KNOB_MASS)):
        p = f"/Part/{body}"
        prim = tp._xform(stage, p)
        paths[body] = p
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
        tp._apply(prim, "PhysxRigidBodyAPI", {
            "physxRigidBody:solverPositionIterationCount": 32,
            "physxRigidBody:solverVelocityIterationCount": 4,
            "physxRigidBody:maxDepenetrationVelocity": 0.3,
        })
    # the knob's link sits on the dial's axis at the face
    from pxr import UsdGeom

    UsdGeom.Xformable(stage.GetPrimAtPath(paths["knob"])).AddTranslateOp().Set(Gf.Vec3d(DIAL[0], DIAL[1], FACE_TOP))
    tp._write_shapes(stage, KIND, paths, looks, mats)
    # markings and the display's strokes (looks only)
    for i, (x0, y0, x1, y1, w) in enumerate(marking_strokes()):
        d = np.array([x1 - x0, y1 - y0])
        n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9) * w / 2
        e = d / max(np.linalg.norm(d), 1e-9) * w / 2
        a, b = np.array([x0, y0]) - e, np.array([x1, y1]) + e
        _quad(stage, f"/Part/case/mark_{i}", [a - n, b - n, b + n, a + n], FACE_TOP + 0.1 * MM, looks["marking"])
    for name, (u0, u1, v0, v1) in display_cells().items():
        corners = [_uv_to_xy(u, v, (LCD_MID_X, 0.0)) for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]
        _quad(stage, f"/Part/case/seg_{name.replace('.', '_')}", corners, FACE_TOP + 0.5 * MM, looks["segment"],
              movable=True)
    base_prim = stage.GetPrimAtPath(paths["case"])
    UsdPhysics.ArticulationRootAPI.Apply(base_prim)
    tp._apply(base_prim, "PhysxArticulationAPI", {
        "physxArticulation:enabledSelfCollisions": False,
        "physxArticulation:solverPositionIterationCount": 32,
        "physxArticulation:solverVelocityIterationCount": 4,
    })
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Part/joints/selector")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(paths["case"])])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(paths["knob"])])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(min(POSITIONS.values()))
    joint.CreateUpperLimitAttr(max(POSITIONS.values()))
    joint.CreateLocalPos0Attr(Gf.Vec3f(DIAL[0], DIAL[1], FACE_TOP))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(math.radians(SPRING))
    drive.CreateDampingAttr(math.radians(DAMPING))
    drive.CreateTargetPositionAttr(POSITIONS["OFF"])
    tp._apply(joint.GetPrim(), "PhysxJointAPI", {"physxJoint:armature": ARMATURE})
    stage.GetRootLayer().Save()


def arena_object(spec: tp.Part):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_physx.assets.articulation import Articulation as PhysxArticulation

    off = math.radians(POSITIONS["OFF"])

    class _MeterArticulation(PhysxArticulation):
        def _to(self, angle: float):
            import torch

            self.set_joint_position_target(torch.full((self.num_instances, self.num_joints), angle,
                                                      device=self.device))

        def reset(self, env_ids=None, env_mask=None):
            # the selector back to OFF, at rest
            import torch

            super().reset(env_ids, env_mask)
            q = torch.full((self.num_instances, self.num_joints), off, device=self.device)
            self.write_joint_position_to_sim_index(position=q)
            self.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
            self._to(off)

        def set_joint_position_target_index(self, *, target, joint_ids=None, env_ids=None, full_data=False):
            # only reset events write targets this way (they would set the target to the default)
            super().set_joint_position_target_index(target=target, joint_ids=joint_ids, env_ids=env_ids,
                                                    full_data=full_data)
            self._to(off)

    class _Meter(Object):
        def _generate_articulation_cfg(self):
            cfg = super()._generate_articulation_cfg()
            cfg.class_type = _MeterArticulation
            cfg.actuators = {"selector": ImplicitActuatorCfg(joint_names_expr=["selector"], stiffness=SPRING,
                                                             damping=DAMPING, armature=ARMATURE)}
            cfg.init_state.joint_pos = {"selector": off}
            return cfg

    cfg = sim_utils.UsdFileCfg(usd_path=tp.usd_file(KIND), fix_root_link=True)
    pose = Pose(position_xyz=tuple(spec.pos), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    return _Meter(name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=cfg, initial_pose=pose)


def selector_deg(base, name: str) -> float:
    meter = base.scene.articulations[name]
    q = meter.data.joint_pos.torch[0].detach().cpu().numpy().astype(np.float64)
    return float(math.degrees(q[list(meter.joint_names).index("selector")]))


def position(deg: float) -> str | None:
    """The selector position whose contacts are made at ``deg`` (None between positions)."""
    for name, d in POSITIONS.items():
        if abs(deg - d) <= IN_POSITION_DEG:
            return name
    return None


def update(base, name: str, states: dict) -> None:
    """Once per control step: aim the selector's spring at the nearest position (the click)."""
    import torch

    meter = base.scene.articulations[name]
    deg = selector_deg(base, name)
    nearest = min(POSITIONS.values(), key=lambda d: abs(d - deg))
    meter.set_joint_position_target(torch.full((1, meter.num_joints), math.radians(nearest), device=base.device))


@dataclass(frozen=True)
class MeterState(task_file.ObjectState):
    """The meter's ``ObjectState`` plus its selector's angle (deg) and position (None between positions)."""

    selector_deg: float = 0.0
    selector: str | None = None


def state(base, name: str) -> MeterState:
    meter = base.scene.articulations[name]
    origin = base.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
    pose = meter.data.root_pose_w.torch[0].detach().cpu().numpy().astype(np.float64)
    pos = pose[:3] - origin
    lo, hi = tp.bounds(KIND)
    deg = selector_deg(base, name)
    return MeterState(pos=pos + np.array(tp.local_center(KIND)), quat_xyzw=pose[3:], lin_vel=np.zeros(3),
                      ang_vel=np.zeros(3), low=pos + np.array(lo), high=pos + np.array(hi), selector_deg=deg,
                      selector=position(deg))


# -- showing the reading -----------------------------------------------------------------------------------------
_PRIMS: dict = {"cells": None, "shown": None}


def show(text: str) -> None:
    """Show ``text`` on the display: each segment to show is raised to the LCD's surface, every other one lowered
    under it. (Moving them, not toggling their visibility, keeps the renderer's frame history from smearing an old
    reading into the new one.) Only segments that change are touched."""
    from pxr import Gf, UsdGeom

    if _PRIMS["cells"] is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        cells = {}
        for prim in stage.Traverse():
            n = prim.GetName()
            if n.startswith("seg_") and "/case/" in str(prim.GetPath()):
                ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
                cells[n[4:].replace("_", ".")] = ops[0]
        _PRIMS["cells"], _PRIMS["shown"] = cells, None
    on = text_cells(text)
    shown = _PRIMS["shown"]
    for key, op in _PRIMS["cells"].items():
        vis = key in on
        if shown is None or (key in shown) != vis:
            op.Set(Gf.Vec3d(0.0, 0.0, SHOWN_LIFT if vis else -HIDDEN_DROP))
    _PRIMS["shown"] = on
