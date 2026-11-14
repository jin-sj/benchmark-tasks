"""A powered bench DIN-rail cutter: a fixed base (an infeed ledge with guide fences, a die the rail passes through,
an outfeed ledge with a length scale), a cutting head over the die (its blade on a slide inside a guard), a toggle
hold-down clamp on the infeed side, and a cut button on a console at the front.

Frame (the cutter's): origin on the table at the blade's plane, under the middle of the die; y along the rail's
path (+y the infeed side, the robot's left; -y the outfeed side, where the cut length is measured), x across it (+x
away from the arms), z up. The blade's plane is y = 0.

The clamp is a hinged arm: a handle toward the arms, a pad over the rail near its hinge (behind the rail). It is a
toggle clamp, sprung over centre: raised (``UP_DEG``) it stays open; pushed down past ``OVER_DEG`` it snaps shut and
its pad holds the rail down on the infeed ledge. Lifted back past ``OVER_DEG`` it springs open.

The cut: pressing the button (``PRESS_MM`` in) runs one cutting stroke if the clamp is shut (an interlock: with the
clamp open the button does nothing): the blade comes down through the die in ``DOWN_S``, and goes back up. The cut is
made when the blade reaches the bottom of its stroke. The blade is a look only: nothing is cut in the simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import tb_parts as tp
from yam_arena import task_file

MM = 1e-3
KIND = "rail_cutter"

# -- base (cutter frame, metres) --------------------------------------------------------------------------------------
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
SCALE_TO = 300 * MM  # the scale's marks run from the blade to here along the outfeed (-y)
HEAD = (-32 * MM, 32 * MM, -22 * MM, 22 * MM, 110 * MM)  # the cutting head's housing over the die: x0, x1, y0, y1, height

# -- the clamp (a toggle clamp on the infeed side, behind the die) ------------------------------------------------------
CLAMP_Y = 45 * MM  # along the rail, from the blade's plane
PIVOT = (40 * MM, LEDGE_TOP + 40 * MM)  # its hinge axis (x, z), along y
ARM = (205 * MM, 16 * MM, 12 * MM)  # its arm: length toward the arms from the pivot (+5 mm behind), width, thickness
HANDLE = (80 * MM, 22 * MM)  # the handle at the arm's end: length, square section
PAD = (-20 * MM, 20 * MM, 30 * MM)  # its pad over the rail: x0, x1 (across the rail), length along it
PAD_BITE = 0.5 * MM  # shut (the arm level), the pad's face would sit this far below the rail's lips' tops
UP_DEG = 45.0  # open: raised to its upper stop
OVER_DEG = 20.0  # its over-centre point
SHUT_DEG = 5.0  # shut: within this of level (on the rail)
SPRING = 1.5  # N m/rad: toward OPEN_TARGET above OVER_DEG, toward SHUT_TARGET (holding the pad down) below it
OPEN_TARGET = 60.0
SHUT_TARGET = -10.0
LEVER_MASS = 0.3
DAMPING = 0.05
ARMATURE = 1e-4

# -- the cut button (on a console at the front, outfeed side) -----------------------------------------------------------
CONSOLE = (-105 * MM, -65 * MM, -75 * MM, -35 * MM, 60 * MM)  # x0, x1, y0, y1, height
BUTTON = (20 * MM, 6 * MM)  # its cap: square side, how far it stands up
PRESS_MM = 3.0  # pressed this far in, it starts a stroke (it can go 5 mm)
BUTTON_TRAVEL = 5 * MM
BUTTON_SPRING = 1000.0  # N/m, preloaded 2 mm against its stop: about 5 N to press it in 3 mm
BUTTON_PRELOAD = 2 * MM

# -- the blade's stroke ---------------------------------------------------------------------------------------------
STROKE = 50 * MM  # the blade comes down this far (from above the die's opening to below the rail)
DOWN_S = 1.0  # a stroke: down in this long ...
UP_S = 0.5  # ... and back up in this long

LOOKS = {"cutter": ((0.20, 0.36, 0.62), 0.5, 0.2), "grip": ((0.85, 0.15, 0.10), 0.6, 0.0),
         "blade": ((0.80, 0.82, 0.85), 0.25, 0.9), "mark": ((0.05, 0.05, 0.06), 0.8, 0.0),
         "head": ((0.25, 0.27, 0.30), 0.5, 0.3), "guard": ((0.95, 0.75, 0.10), 0.5, 0.0),
         "button": ((0.85, 0.10, 0.08), 0.4, 0.0), "pad": ((0.10, 0.10, 0.11), 0.8, 0.0)}


def handle_point(deg: float) -> np.ndarray:
    """The middle of the clamp's handle (x, z) in the cutter frame with the clamp at ``deg``."""
    a = math.radians(deg)
    r = ARM[0] - 5 * MM - HANDLE[0] / 2  # from the pivot to the handle's middle
    return np.array([PIVOT[0] - r * math.cos(a), PIVOT[1] + r * math.sin(a)])


def button_top() -> np.ndarray:
    """The middle of the button's cap's top (x, y, z), at rest, in the cutter frame."""
    x0, x1, y0, y1, h = CONSOLE
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2, h + BUTTON[1]])


def cutter_shapes() -> list[tp.Shape]:
    b = lambda *v, look="cutter", mat="steel": tp._box("base", look, mat, *v)  # noqa: E731
    hx, ax, ah = LEDGE_HALF_X, APERTURE[0] / 2, LEDGE_TOP + APERTURE[1]
    out = [
        b(-hx, hx, *INFEED, 0.0, LEDGE_TOP), b(-hx, hx, *OUTFEED, 0.0, LEDGE_TOP),  # the ledges
        b(-ax, ax, -DIE_Y, DIE_Y, 0.0, LEDGE_TOP),  # the die's floor
        b(-hx - 1 * MM, -ax, -DIE_Y, DIE_Y - LEAD_IN[-1][0], 0.0, DIE_TOP),  # the die's cheeks
        b(ax, hx + 1 * MM, -DIE_Y, DIE_Y - LEAD_IN[-1][0], 0.0, DIE_TOP),
        b(-ax, ax, -DIE_Y, DIE_Y - LEAD_IN[-1][0], ah, DIE_TOP),  # the die's top
    ]
    fx, ft, fh = FENCE  # the guide fences along the infeed ledge, either side of the rail
    out += [b(-fx - ft, -fx, DIE_Y, INFEED[1], LEDGE_TOP, LEDGE_TOP + fh),
            b(fx, fx + ft, DIE_Y, INFEED[1], LEDGE_TOP, LEDGE_TOP + fh)]
    d0 = 0.0  # the die's mouth on the infeed side: a stepped lead-in, the opening widening toward the infeed
    for d1, grow in LEAD_IN:
        y0, y1, d0 = DIE_Y - d1, DIE_Y - d0, d1
        wx, top = ax + grow, ah + grow
        out += [b(-hx - 1 * MM, -wx, y0, y1, 0.0, DIE_TOP), b(wx, hx + 1 * MM, y0, y1, 0.0, DIE_TOP),
                b(-wx, wx, y0, y1, top, DIE_TOP)]
    # the cutting head over the die (motor and blade housing), a yellow guard band round it
    x0, x1, y0, y1, hh = HEAD
    out += [b(x0, x1, y0, y1, DIE_TOP, DIE_TOP + hh, look="head"),
            b(x0 - 1 * MM, x1 + 1 * MM, y0 - 1 * MM, y1 + 1 * MM, DIE_TOP, DIE_TOP + 25 * MM, look="guard", mat=None)]
    # the clamp's post, behind the rail
    out.append(b(hx + 1 * MM, PIVOT[0] + 12 * MM, CLAMP_Y - 12 * MM, CLAMP_Y + 12 * MM, 0.0, PIVOT[1] + 10 * MM))
    # the console with the cut button
    cx0, cx1, cy0, cy1, ch = CONSOLE
    out.append(b(cx0, cx1, cy0, cy1, 0.0, ch, look="head"))
    # the length scale on the outfeed ledge's top, on the near side of the rail: a mark every 10 mm, longer every 50
    for i in range(int(round(SCALE_TO / (10 * MM))) + 1):
        y = -i * 10 * MM
        long = 10 if i % 10 == 0 else (7 if i % 5 == 0 else 4)
        out.append(b(-hx, -hx + long * MM, y - 0.4 * MM, y + 0.4 * MM, LEDGE_TOP, LEDGE_TOP + 0.2 * MM,
                     look="mark", mat=None))
    # the clamp, drawn shut (level): arm, handle, pad over the rail
    px, pz = PIVOT
    lv = lambda *v, look="cutter", mat="steel": tp._box("lever", look, mat, *v)  # noqa: E731
    x_end = px - (ARM[0] - 5 * MM)
    cy, aw = CLAMP_Y, ARM[1] / 2
    out += [
        lv(x_end, px + 5 * MM, cy - aw, cy + aw, pz - ARM[2] / 2, pz + ARM[2] / 2),
        lv(x_end, x_end + HANDLE[0], cy - HANDLE[1] / 2, cy + HANDLE[1] / 2, pz - HANDLE[1] / 2, pz + HANDLE[1] / 2,
           look="grip", mat="rubber"),
        lv(PAD[0], PAD[1], cy - PAD[2] / 2, cy + PAD[2] / 2, LEDGE_TOP + 7.5 * MM - PAD_BITE, pz - ARM[2] / 2,
           look="pad", mat="rubber"),
    ]
    # the button's cap (at rest)
    bt = button_top()
    s = BUTTON[0] / 2
    out.append(tp._box("button", "button", "plastic", bt[0] - s, bt[0] + s, bt[1] - s, bt[1] + s, CONSOLE[4] - 4 * MM,
                       bt[2]))
    # the blade (a look only), raised: its edge above the die's opening
    out.append(tp._box("blade", "blade", None, -ax, ax, -0.75 * MM, 0.75 * MM, LEDGE_TOP - 2 * MM + STROKE,
                       LEDGE_TOP + 40 * MM + STROKE))
    return out


def install() -> None:
    tp.SHAPES[KIND] = cutter_shapes
    tp.LOOKS.update(LOOKS)
    tp.CUSTOM_KINDS[KIND] = (write_usd, arena_object, state, update)


def part(name: str, pos) -> tp.Part:
    lo, hi = tp.bounds(KIND)
    size = tuple(b - a for a, b in zip(lo, hi, strict=True))
    return tp.Part(name=name, pos=tuple(float(v) for v in pos), size=size, mass=None, color=LOOKS["cutter"][0], kind=KIND)


# -- USD -------------------------------------------------------------------------------------------------------------
JOINTS = ("lever", "button", "blade")


def write_usd(path) -> None:
    from pxr import Gf, Sdf, UsdPhysics

    stage, root, looks, mats = tp._new_stage(path)
    paths = {}
    masses = {"base": 5.0, "lever": LEVER_MASS, "button": 0.01, "blade": 0.1}
    for body in ("base", "lever", "button", "blade"):
        p = f"/Part/{body}"
        prim = tp._xform(stage, p)
        paths[body] = p
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(masses[body])
        if body == "blade":  # no colliders: give it an inertia
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-4, 1e-4, 1e-4))
            mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, LEDGE_TOP + 20 * MM + STROKE))
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

    def joint(name, kind, axis, lo, hi, at, stiffness, damping, target):
        j = (UsdPhysics.RevoluteJoint if kind == "angular" else UsdPhysics.PrismaticJoint).Define(
            stage, f"/Part/joints/{name}")
        j.CreateBody0Rel().SetTargets([Sdf.Path(paths["base"])])
        j.CreateBody1Rel().SetTargets([Sdf.Path(paths[name])])
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
        drive.CreateTargetPositionAttr(target)
        tp._apply(j.GetPrim(), "PhysxJointAPI", {"physxJoint:armature": ARMATURE})

    # the clamp: about +y a positive angle raises its handle (its -x end)
    joint("lever", "angular", "Y", 0.0, UP_DEG, (PIVOT[0], CLAMP_Y, PIVOT[1]), math.radians(SPRING),
          math.radians(DAMPING), OPEN_TARGET)
    bt = button_top()
    joint("button", "linear", "Z", -BUTTON_TRAVEL, 0.0, tuple(bt), BUTTON_SPRING, 5.0, BUTTON_PRELOAD)
    joint("blade", "linear", "Z", -STROKE, 0.0, (0.0, 0.0, LEDGE_TOP + STROKE), 20000.0, 400.0, 0.0)
    stage.GetRootLayer().Save()


# -- the clamp's spring, the button, the stroke (once per control step) -------------------------------------------------
_RUN: dict = {}  # per articulation: the stroke's state


def _targets(art, clamp_deg: float, blade: float) -> None:
    import torch

    t = torch.zeros((art.num_instances, art.num_joints), device=art.device)
    names = list(art.joint_names)
    t[:, names.index("lever")] = math.radians(OPEN_TARGET if clamp_deg > OVER_DEG else SHUT_TARGET)
    t[:, names.index("button")] = BUTTON_PRELOAD
    t[:, names.index("blade")] = blade
    art.set_joint_position_target(t)


def _joints(art) -> dict:
    q = art.data.joint_pos.torch[0].detach().cpu().numpy().astype(np.float64)
    return {n: float(q[i]) for i, n in enumerate(art.joint_names)}


def _new_run() -> dict:
    return {"start": None, "armed": True, "strokes": 0, "step": -1}


def update(base, name: str, out) -> None:
    """Once per control step: the clamp's over-centre spring, and the stroke (started by the button, if the clamp is
    shut; the button must come back up before it starts another)."""
    art = base.scene.articulations[name]
    run = _RUN.setdefault(id(art), _new_run())
    step = int(base.episode_length_buf[0])
    if step < run["step"] or step <= 1:
        run.clear()
        run.update(_new_run())
    if step == run["step"]:  # asked again within the same control step
        return
    run["step"] = step
    j = _joints(art)
    clamp_deg = math.degrees(j["lever"])
    pressed = -j["button"] * 1000.0 >= PRESS_MM
    if not pressed and -j["button"] * 1000.0 < 1.0:
        run["armed"] = True
    if run["start"] is None and pressed and run["armed"] and clamp_deg <= SHUT_DEG:
        run["start"], run["armed"] = step, False
        run["strokes"] += 1
    blade = 0.0
    if run["start"] is not None:
        t = (step - run["start"]) * float(base.step_dt)
        if t <= DOWN_S:
            blade = -STROKE * t / DOWN_S
        elif t <= DOWN_S + UP_S:
            blade = -STROKE * (1.0 - (t - DOWN_S) / UP_S)
        else:
            run["start"] = None
    _targets(art, clamp_deg, blade)


def arena_object(spec: tp.Part):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    from isaaclab_physx.assets.articulation import Articulation as PhysxArticulation

    up = math.radians(UP_DEG)

    class _CutterArticulation(PhysxArticulation):
        def reset(self, env_ids=None, env_mask=None):
            # the clamp open, the button and the blade up, at rest
            import torch

            super().reset(env_ids, env_mask)
            q = torch.zeros((self.num_instances, self.num_joints), device=self.device)
            q[:, list(self.joint_names).index("lever")] = up
            self.write_joint_position_to_sim_index(position=q)
            self.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
            _RUN.pop(id(self), None)
            _targets(self, UP_DEG, 0.0)

        def set_joint_position_target_index(self, *, target, joint_ids=None, env_ids=None, full_data=False):
            # only reset events write targets this way (they would set the targets to the defaults)
            super().set_joint_position_target_index(target=target, joint_ids=joint_ids, env_ids=env_ids,
                                                    full_data=full_data)
            _targets(self, UP_DEG, 0.0)

    class _Cutter(Object):
        def _generate_articulation_cfg(self):
            cfg = super()._generate_articulation_cfg()
            cfg.class_type = _CutterArticulation
            cfg.actuators = {
                "lever": ImplicitActuatorCfg(joint_names_expr=["lever"], stiffness=SPRING, damping=DAMPING,
                                             armature=ARMATURE),
                "button": ImplicitActuatorCfg(joint_names_expr=["button"], stiffness=BUTTON_SPRING, damping=5.0,
                                              armature=ARMATURE),
                "blade": ImplicitActuatorCfg(joint_names_expr=["blade"], stiffness=20000.0, damping=400.0,
                                             armature=ARMATURE),
            }
            cfg.init_state.joint_pos = {"lever": up, "button": 0.0, "blade": 0.0}
            return cfg

    cfg = sim_utils.UsdFileCfg(usd_path=tp.usd_file(KIND), fix_root_link=True)
    pose = Pose(position_xyz=tuple(spec.pos), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    return _Cutter(name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=cfg, initial_pose=pose)


@dataclass(frozen=True)
class CutterState(task_file.ObjectState):
    """The cutter's ``ObjectState`` (its origin) plus its clamp's angle (degrees; 0 level, on the rail), how far its
    button is pressed in (mm), how far its blade is down (mm, 0..STROKE) and how many strokes it has made."""

    clamp_deg: float = UP_DEG
    button_mm: float = 0.0
    blade_mm: float = 0.0
    strokes: int = 0


def state(base, name: str) -> CutterState:
    art = base.scene.articulations[name]
    origin = base.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
    pose = art.data.root_pose_w.torch[0].detach().cpu().numpy().astype(np.float64)
    pos = pose[:3] - origin
    j = _joints(art)
    lo, hi = tp.bounds(KIND)
    return CutterState(pos=pos + np.array(tp.local_center(KIND)), quat_xyzw=pose[3:], lin_vel=np.zeros(3),
                       ang_vel=np.zeros(3), low=pos + np.array(lo), high=pos + np.array(hi),
                       clamp_deg=math.degrees(j["lever"]), button_mm=-j["button"] * 1000.0,
                       blade_mm=-j["blade"] * 1000.0, strokes=_RUN.get(id(art), {}).get("strokes", 0))
