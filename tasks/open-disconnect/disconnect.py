"""A strip of knife-disconnect terminal blocks: each block carries a hinged lever that opens or closes its circuit.

The strip is one articulation with a fixed base (the blocks' housings, latched on the rail: they do not move) and one
hinged lever per block. A lever lies flat along the block's top when closed, its free end toward the arms, with an
upright tab at that end; opened, it stands upright (95 degrees). It snaps between the two: a spring pulls it toward
whichever end is nearer (it changes side at MID_DEG), against a stop at each end.

Frames: the strip's origin is on the rail's lip-top plane, on the rail's centreline, at the middle of the strip;
x across the rail (away from the arms), y along it (the robot's left), z up. Block i (0 = leftmost) is centred at
``block_y(i)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import tb_parts as tp
from yam_arena import task_file

MM = 1e-3
KIND = "disconnect_strip"

# -- geometry (strip frame, metres) ---------------------------------------------------------------------------
N_BLOCKS = 5
WIDTH = 6.2 * MM  # a block's thickness along the rail (a 4 mm2 disconnect terminal)
SEAM = 0.15 * MM  # each housing drawn this much short of its faces, so the blocks read as separate
TOP = 34 * MM  # the housing's top (where the lever lies) above the lip tops
# the housing's side view (x, z), counter-clockwise: stepped wings with the wire entries, feet beside the rail's lips
PROFILE = [(-31, -4.0), (-27, -6.0), (-19, -6.0), (-18.2, 0.3), (18.2, 0.3), (19, -6.0), (27, -6.0), (31, -4.0),
           (31, 20), (27, 20), (27, 26), (20, 26), (20, 34), (-20, 34), (-20, 26), (-27, 26), (-27, 20), (-31, 20)]
BOXES = [(-31, 31, 0.3, 20), (-27, 27, 20, 26), (-20, 20, 26, 34)]  # the housing's collision (x0, x1, z0, z1), mm
PIVOT = (2 * MM, TOP + 1.5 * MM)  # the hinge axis (x, z): along the rail, just above the housing's top
ARM = (16 * MM, 3 * MM)  # the lever's arm: from the hinge toward the arms (closed), and its thickness
TAB = (3 * MM, 6 * MM)  # the upright tab at the arm's free end: thickness (across), height above the arm
LEVER_WIDTH = 4.0 * MM  # along the rail
LEVER_MASS = 0.0006

# -- the snap ----------------------------------------------------------------------------------------------------
OPEN_DEG = 95.0  # the open stop
MID_DEG = 45.0  # below this the spring pulls the lever closed, above it open
PRELOAD_DEG = 12.0  # the spring's target lies this far beyond each stop: the lever is held against the stop
SPRING = 0.02  # N m/rad: at the middle about 0.02 N m, i.e. ~1.5-2.5 N on the lever
DAMPING = 0.0004  # N m s/rad
ARMATURE = 2e-6  # kg m2

LOOKS = {"lever": ((0.93, 0.45, 0.10), 0.5, 0.0)}


def block_y(i: int) -> float:
    return ((N_BLOCKS - 1) / 2 - i) * WIDTH


def strip_shapes() -> list[tp.Shape]:
    out = []
    px, pz = PIVOT
    for i in range(N_BLOCKS):
        y = block_y(i)
        y0, y1 = y - WIDTH / 2, y + WIDTH / 2
        out.append(tp._prism("base", "grey", None, [(x * MM, z * MM) for x, z in PROFILE], y0 + SEAM, y1 - SEAM))
        for x0, x1, z0, z1 in BOXES:
            b = tp._box("base", "grey", "polyamide", x0 * MM, x1 * MM, y0, y1, z0 * MM, z1 * MM)
            out.append(tp.Shape(b.body, b.look, b.material, b.center, b.size, visible=False))
        l0, l1 = y - LEVER_WIDTH / 2, y + LEVER_WIDTH / 2
        body = f"lever_{i}"
        out.append(tp._box(body, "lever", "polyamide", px - ARM[0], px + 2 * MM, l0, l1, pz - ARM[1] / 2, pz + ARM[1] / 2))
        out.append(tp._box(body, "lever", "polyamide", px - ARM[0], px - ARM[0] + TAB[0], l0, l1, pz + ARM[1] / 2,
                           pz + ARM[1] / 2 + TAB[1]))
    return out


def install() -> None:
    tp.SHAPES[KIND] = strip_shapes
    tp.LOOKS.update(LOOKS)
    tp.CUSTOM_KINDS[KIND] = (write_usd, arena_object, state, update)


def part(name: str, pos) -> tp.Part:
    """The strip with its origin at ``pos`` (world, m). Fixed base: no mass for yam_arena."""
    lo, hi = tp.bounds(KIND)
    size = tuple(b - a for a, b in zip(lo, hi, strict=True))
    return tp.Part(name=name, pos=tuple(float(v) for v in pos), size=size, mass=None, color=tp.LOOKS["grey"][0], kind=KIND)


def write_usd(path) -> None:
    from pxr import Gf, Sdf, UsdPhysics

    stage, root, looks, mats = tp._new_stage(path)
    bodies = ["base"] + [f"lever_{i}" for i in range(N_BLOCKS)]
    paths = {}
    for body in bodies:
        p = f"/Part/{body}"
        prim = tp._xform(stage, p)
        paths[body] = p
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.05 if body == "base" else LEVER_MASS)
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
    for i in range(N_BLOCKS):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/Part/joints/lever_{i}")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(paths["base"])])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(paths[f"lever_{i}"])])
        joint.CreateAxisAttr("Y")  # about +y a positive angle lifts the lever's free (-x) end
        joint.CreateLowerLimitAttr(0.0)
        joint.CreateUpperLimitAttr(OPEN_DEG)
        pivot = Gf.Vec3f(PIVOT[0], block_y(i), PIVOT[1])
        joint.CreateLocalPos0Attr(pivot)
        joint.CreateLocalPos1Attr(pivot)
        joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(math.radians(SPRING))
        drive.CreateDampingAttr(math.radians(DAMPING))
        drive.CreateTargetPositionAttr(-PRELOAD_DEG)
        tp._apply(joint.GetPrim(), "PhysxJointAPI", {"physxJoint:armature": ARMATURE})
    stage.GetRootLayer().Save()


def _targets(q: np.ndarray) -> np.ndarray:
    """Each lever's spring target (rad) from its angle (rad): beyond the stop on the side it is nearer."""
    return np.where(np.degrees(q) < MID_DEG, math.radians(-PRELOAD_DEG), math.radians(OPEN_DEG + PRELOAD_DEG))


def arena_object(spec: tp.Part):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    from isaaclab_physx.assets.articulation import Articulation as PhysxArticulation

    class _StripArticulation(PhysxArticulation):
        def _closed(self):
            import torch

            self.set_joint_position_target(torch.full((self.num_instances, self.num_joints),
                                                      math.radians(-PRELOAD_DEG), device=self.device))

        def reset(self, env_ids=None, env_mask=None):
            # every lever back to closed, at rest, its spring pulling it closed
            import torch

            super().reset(env_ids, env_mask)
            zero = torch.zeros((self.num_instances, self.num_joints), device=self.device)
            self.write_joint_position_to_sim_index(position=zero)
            self.write_joint_velocity_to_sim_index(velocity=zero)
            self._closed()

        def set_joint_position_target_index(self, *, target, joint_ids=None, env_ids=None, full_data=False):
            # only reset events write targets this way (they would set every target to the default, 0)
            super().set_joint_position_target_index(target=target, joint_ids=joint_ids, env_ids=env_ids,
                                                    full_data=full_data)
            self._closed()

    class _Strip(Object):
        def _generate_articulation_cfg(self):
            cfg = super()._generate_articulation_cfg()
            cfg.class_type = _StripArticulation
            cfg.actuators = {"levers": ImplicitActuatorCfg(joint_names_expr=["lever_.*"], stiffness=SPRING,
                                                           damping=DAMPING, armature=ARMATURE)}
            cfg.init_state.joint_pos = {"lever_.*": 0.0}
            return cfg

    cfg = sim_utils.UsdFileCfg(usd_path=tp.usd_file(KIND), fix_root_link=True)
    pose = Pose(position_xyz=tuple(spec.pos), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    return _Strip(name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=cfg, initial_pose=pose)


@dataclass(frozen=True)
class StripState(task_file.ObjectState):
    """The strip's ``ObjectState`` (its origin) plus, per lever, leftmost block first: its angle now in degrees (0
    closed, 95 open), the largest angle it has reached so far this episode, and for how long (s) it has been open
    (80 degrees or more) without a break, up to now."""

    levers_deg: tuple[float, ...] = ()
    levers_max_deg: tuple[float, ...] = ()
    levers_open_s: tuple[float, ...] = ()


OPEN_RECORD_DEG = 80.0  # a lever at or beyond this counts as open for levers_open_s


@dataclass
class _Run:
    step: int = -1
    max_deg: np.ndarray | None = None
    open_s: np.ndarray | None = None


_RUNS: dict[str, _Run] = {}


def _angles(strip) -> np.ndarray:
    q = strip.data.joint_pos.torch[0].detach().cpu().numpy().astype(np.float64)
    names = list(strip.joint_names)
    return np.array([q[names.index(f"lever_{i}")] for i in range(N_BLOCKS)])


def update(base, name: str, states: dict) -> None:
    """Once per control step: record each lever's largest angle and how long it has been open, and point each
    lever's spring at the stop it is nearer (the snap)."""
    import torch

    strip = base.scene.articulations[name]
    names = list(strip.joint_names)
    q = _angles(strip)
    step = int(base.episode_length_buf[0])
    run = _RUNS.get(name)
    if run is None or step == 0 or step < run.step:  # a new episode
        run = _RUNS[name] = _Run(max_deg=np.degrees(q), open_s=np.zeros(N_BLOCKS))
    if step != run.step:  # once per control step (object_states may be asked more than once a step)
        run.step = step
        deg = np.degrees(q)
        run.max_deg = np.maximum(run.max_deg, deg)
        run.open_s = np.where(deg >= OPEN_RECORD_DEG, run.open_s + float(base.step_dt), 0.0) if step > 0 else run.open_s
    t = _targets(q)
    target = torch.zeros((1, len(names)), device=base.device)
    for i in range(N_BLOCKS):
        target[0, names.index(f"lever_{i}")] = float(t[i])
    strip.set_joint_position_target(target)


def state(base, name: str) -> StripState:
    strip = base.scene.articulations[name]
    run = _RUNS.get(name)
    origin = base.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
    pose = strip.data.root_pose_w.torch[0].detach().cpu().numpy().astype(np.float64)
    pos = pose[:3] - origin
    lo, hi = tp.bounds(KIND)
    c = np.array(tp.local_center(KIND))
    return StripState(pos=pos + c, quat_xyzw=pose[3:], lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                      low=pos + np.array(lo), high=pos + np.array(hi),
                      levers_deg=tuple(float(v) for v in np.degrees(_angles(strip))),
                      levers_max_deg=tuple(float(v) for v in run.max_deg) if run else (),
                      levers_open_s=tuple(float(v) for v in run.open_s) if run else ())
