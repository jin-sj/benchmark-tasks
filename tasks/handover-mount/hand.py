"""A scripted human hand that offers a terminal block and lets go only once the robot has it.

The hand is an articulation with a fixed base: three sprung slides (x, y, z) carry the forearm and hand, and a
fourth moves the thumb across the pinch. Its joint drives are stiff but not rigid (1.5 N/mm), so a robot that
pushes it moves it a little, as it would a person's hand; gravity is off for its links (a person holds their own
arm up).

Frames: the hand's origin is the pinch centre, between the index fingertip pad (+y side) and the thumb pad (-y
side), with the forearm running toward +x (away from the robot). The block is held by its +x end, upright, its
broad faces between the pads.

Script (one control step at a time, from the task's ``object_states`` hook, see :func:`update`):

1. ``reach``: from its start, off the far edge of the table, the hand moves to the offer point over REACH_S.
2. ``offer``: it holds the block there, drifting slowly by a few millimetres (a person does not hold still),
   until the robot grips the block: both robot fingertips pressing the block with GRIP_N or more, for GRIP_S.
3. ``release``: the thumb opens over RELEASE_S; the hand stays put.
4. ``withdraw``: the hand moves back toward its start.

If the robot never grips the block, the hand keeps offering it for the whole episode.

Measured for the scorer, every control step (see :class:`HandState`): the largest contact force between the robot
and the hand so far, the largest load on the hand so far (the net force on it from the robot and the block, as a
force sensor between the hand and its support reads it), whether the hand let go into a robot grip, and whether the
block was dropped. Both forces are checked after every physics step.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

import tb_parts as tp
from yam_arena import task_file

MM = 1e-3

# -- geometry (hand frame, metres) ---------------------------------------------------------------------------
HALF_GAP = tp.BLOCK_WIDTH / 2  # the pads touch the block's broad faces
PAD = (-11 * MM, 14 * MM)  # the fingertip pads' extent along x (the block's end is at x +10 mm)
PAD_Z = (-9 * MM, 9 * MM)
PAD_T = 14 * MM  # a pad's thickness across the pinch
BLOCK_END = 10 * MM  # the block's +x end, in the hand frame
GRIP_AT = (20 * MM, 8 * MM)  # the pinch centre in the block's frame (x along its length, z above its base): low,
# near the feet, so the robot can grip the block's middle above the person's fingertips

SKIN, SLEEVE = "skin", "sleeve"
LOOKS = {SKIN: ((0.86, 0.68, 0.55), 0.6, 0.0), SLEEVE: ((0.30, 0.32, 0.36), 0.8, 0.0)}
MATERIAL = ("skin", (0.8, 0.8, "average"))  # skin; the block's polyamide ("min") governs the pinch: 0.35

# -- dynamics -------------------------------------------------------------------------------------------------
ARM_SPRING, ARM_DAMPING = 1500.0, 120.0  # N/m, N s/m per slide
THUMB_SPRING, THUMB_DAMPING = 400.0, 10.0
PINCH_DEPTH = 7 * MM  # the thumb is driven this far past the block's face: ~2.8 N of pinch
THUMB_OPEN = -25 * MM
MASSES = {"base": 0.2, "sx": 0.05, "sy": 0.05, "arm": 0.6, "thumb": 0.03}
ARMATURE = 0.05

# -- script ----------------------------------------------------------------------------------------------------
REACH_S = 3.0
DRIFT = (3 * MM, 2 * MM, 2 * MM)  # amplitude of the slow drift while offering, per axis
DRIFT_HZ = (0.21, 0.17, 0.13)
GRIP_N = 2.0  # each robot fingertip pressing the block with at least this ...
GRIP_S = 0.3  # ... for this long: the hand lets go
RELEASE_S = 0.4
IN_HAND = 15 * MM  # the block counts as still in the hand while its centre stays this close to where it was held
FALLING = 0.25  # m/s downward with nothing holding it: a drop
GRIPPED_N = 1.0  # each robot fingertip pressing the block with this much: the robot holds it
WITHDRAW_S = 1.5
WITHDRAW = (0.15, 0.0, 0.05)  # metres back from the offer point


def hand_shapes() -> list[tp.Shape]:
    b = tp._box
    out = [
        # index fingertip pad and finger (+y side of the pinch), thumb pad and thumb (-y side, its own body)
        b("arm", SKIN, "skin", PAD[0], PAD[1], HALF_GAP, HALF_GAP + PAD_T, *PAD_Z),
        b("thumb", SKIN, "skin", PAD[0], PAD[1], -HALF_GAP - PAD_T, -HALF_GAP, *PAD_Z),
        # the other three fingers, curled under the index finger, clear of the block
        b("arm", SKIN, "skin", 2 * MM, 16 * MM, HALF_GAP + 1 * MM, HALF_GAP + 22 * MM, -32 * MM, -11 * MM),
        # palm, beyond the block's end
        b("arm", SKIN, "skin", 16 * MM, 40 * MM, -26 * MM, 30 * MM, -34 * MM, 26 * MM),
        # wrist and forearm (the forearm in a sleeve)
        b("arm", SKIN, "skin", 40 * MM, 110 * MM, -24 * MM, 24 * MM, -26 * MM, 20 * MM),
        b("arm", SLEEVE, "skin", 110 * MM, 300 * MM, -32 * MM, 32 * MM, -32 * MM, 26 * MM),
    ]
    return out


def install() -> None:
    """Teach tb_parts about the hand part (shapes, looks, material, spawning and per-step state)."""
    tp.SHAPES["hand"] = hand_shapes
    tp.LOOKS.update(LOOKS)
    tp.MATERIALS[MATERIAL[0]] = MATERIAL[1]
    tp.CUSTOM_KINDS["hand"] = (write_usd, arena_object, state, update)


def part(name: str, pos) -> tp.Part:
    """The hand at its start pose: ``pos`` is the pinch centre (world, m). Fixed base: no mass for yam_arena."""
    lo, hi = tp.bounds("hand")
    size = tuple(b - a for a, b in zip(lo, hi, strict=True))
    return tp.Part(name=name, pos=tuple(pos), size=size, mass=None, color=LOOKS[SKIN][0], kind="hand")


# -- USD ---------------------------------------------------------------------------------------------------------


def write_usd(path) -> None:
    from pxr import Gf, Sdf, UsdPhysics

    stage, root, looks, mats = tp._new_stage(path)
    bodies = ["base", "sx", "sy", "arm", "thumb"]
    paths = {}
    for body in bodies:
        p = f"/Part/{body}"
        prim = tp._xform(stage, p)
        paths[body] = p
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(MASSES[body])
        tp._apply(prim, "PhysxRigidBodyAPI", {
            "physxRigidBody:disableGravity": True,
            "physxRigidBody:solverPositionIterationCount": 32,
            "physxRigidBody:solverVelocityIterationCount": 4,
            "physxRigidBody:maxDepenetrationVelocity": 0.3,
        })
        if body in ("arm", "thumb"):  # contact sensing: the robot touching the hand
            tp._apply(prim, "PhysxContactReportAPI", {"physxContactReport:threshold": 0.0})
    tp._write_shapes(stage, "hand", paths, looks, mats)
    base_prim = stage.GetPrimAtPath(paths["base"])  # the root on a rigid body, so Isaac Lab can fix it to the world
    UsdPhysics.ArticulationRootAPI.Apply(base_prim)
    tp._apply(base_prim, "PhysxArticulationAPI", {
        "physxArticulation:enabledSelfCollisions": False,
        "physxArticulation:solverPositionIterationCount": 32,
        "physxArticulation:solverVelocityIterationCount": 4,
    })
    joints = (  # name, parent, child, axis, limits, spring, damping
        ("hand_x", "base", "sx", "X", (-0.6, 0.6), ARM_SPRING, ARM_DAMPING),
        ("hand_y", "sx", "sy", "Y", (-0.6, 0.6), ARM_SPRING, ARM_DAMPING),
        ("hand_z", "sy", "arm", "Z", (-0.4, 0.4), ARM_SPRING, ARM_DAMPING),
        ("thumb", "arm", "thumb", "Y", (THUMB_OPEN - 5 * MM, PINCH_DEPTH + 2 * MM), THUMB_SPRING, THUMB_DAMPING),
    )
    for name, parent, child, axis, (lo, hi), k, c in joints:
        joint = UsdPhysics.PrismaticJoint.Define(stage, f"/Part/joints/{name}")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(paths[parent])])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(paths[child])])
        joint.CreateAxisAttr(axis)
        joint.CreateLowerLimitAttr(lo)
        joint.CreateUpperLimitAttr(hi)
        joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(k)
        drive.CreateDampingAttr(c)
        drive.CreateTargetPositionAttr(0.0)
        tp._apply(joint.GetPrim(), "PhysxJointAPI", {"physxJoint:armature": ARMATURE})
    stage.GetRootLayer().Save()


def arena_object(spec: tp.Part):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    from isaaclab_physx.assets.articulation import Articulation as PhysxArticulation

    class _HandArticulation(PhysxArticulation):
        """Resets with the thumb already squeezing: the block must stay pinched through the reset's settle steps,
        which run before this task's per-step hook does."""

        def __init__(self, cfg):
            # Isaac Lab turns PhysX contact processing off unless a contact sensor asks for it; the hand's contact
            # views need it. Set here, as Isaac Lab's ContactSensor does: during scene construction, after the
            # physics manager's own settings and before the simulation starts.
            from isaaclab.app.settings_manager import get_settings_manager

            get_settings_manager().set_bool("/physics/disableContactProcessing", False)
            super().__init__(cfg)

        def _squeeze(self):
            import torch

            idx = list(self.joint_names).index("thumb")
            self.set_joint_position_target(torch.full((self.num_instances, 1), PINCH_DEPTH, device=self.device),
                                           joint_ids=[idx])

        def reset(self, env_ids=None, env_mask=None):
            # The environment's reset puts the block back at its start but leaves this hand's joints where the last
            # episode left them; springing back, the hand would sweep into the block and fling it. So: the joints
            # back to the start, at rest, the slides' targets there too, and the thumb squeezing.
            import torch

            super().reset(env_ids, env_mask)
            default = self.data.default_joint_pos
            default = (default.torch if hasattr(default, "torch") else default).clone()
            self.write_joint_position_to_sim_index(position=default)
            self.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(default))
            self.set_joint_position_target(default)
            self._squeeze()

        def set_joint_position_target_index(self, *, target, joint_ids=None, env_ids=None, full_data=False):
            # Only reset events write targets this way (the script uses set_joint_position_target): an event that
            # resets every joint target to its default runs after reset() and would leave the thumb slack
            super().set_joint_position_target_index(target=target, joint_ids=joint_ids, env_ids=env_ids,
                                                    full_data=full_data)
            self._squeeze()

    class _Hand(Object):
        def _generate_articulation_cfg(self):
            cfg = super()._generate_articulation_cfg()
            cfg.class_type = _HandArticulation
            cfg.actuators = {
                "arm": ImplicitActuatorCfg(joint_names_expr=["hand_.*"], stiffness=ARM_SPRING, damping=ARM_DAMPING,
                                           armature=ARMATURE),
                "thumb": ImplicitActuatorCfg(joint_names_expr=["thumb"], stiffness=THUMB_SPRING,
                                             damping=THUMB_DAMPING, armature=ARMATURE),
            }
            cfg.init_state.joint_pos = {"hand_.*": 0.0, "thumb": 0.0}
            return cfg

    cfg = sim_utils.UsdFileCfg(usd_path=tp.usd_file("hand"), fix_root_link=True)
    pose = Pose(position_xyz=tuple(spec.pos), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    return _Hand(name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=cfg, initial_pose=pose)


# -- the script and the measurements --------------------------------------------------------------------------


@dataclass(frozen=True)
class HandState(task_file.ObjectState):
    """The hand's ``ObjectState`` (pose of its pinch centre) plus what the scorer needs from the episode so far."""

    phase: str = "reach"
    released: bool = False  # the hand let go of the block into a robot grip
    peak_contact_n: float = 0.0  # the largest robot-hand contact force so far (N)
    peak_load_n: float = 0.0  # the largest load on the hand so far (N): the net force on it from the robot and the block
    load_n: float = 0.0  # this step's load on the hand (N)
    contact_n: float = 0.0  # this step's robot-hand contact force (N)
    grip_n: tuple[float, float] = (0.0, 0.0)  # each robot fingertip pressing the block now (N)
    touch_n: float = 0.0  # all robot fingertips on the block now, added up (N): 0 once the robot has let go
    snatched: bool = False  # the block left the hand before the hand let go (pulled or knocked out)
    dropped: bool = False  # the block fell (faster than 0.25 m/s) with nothing holding it


@dataclass
class _Run:
    step: int = -1
    phase: str = "reach"
    grip_steps: int = 0
    phase_t0: float = 0.0
    released: bool = False
    peak: float = 0.0
    contact: float = 0.0
    grip: tuple[float, float] = (0.0, 0.0)
    offer_q: np.ndarray | None = None
    views: object = None
    callback: object = None  # the per-physics-step handle (kept alive here)
    substep_peak: float = 0.0  # the largest robot-hand force over the physics steps since the last control step
    load_substep_peak: float = 0.0  # the largest load on the hand over the physics steps since the last control step
    load: float = 0.0
    load_peak: float = 0.0
    touch: float = 0.0
    substeps: int = 0
    last_substeps: int = 0
    held_offset: np.ndarray | None = None  # the block's centre relative to the pinch while the hand holds it
    snatched: bool = False
    dropped: bool = False


# Configured by the task: which block the hand holds, the offer point, and which robot links count.
CONFIG = {"block": None, "offer": None, "table_top": None, "robots": ("left_robot", "right_robot"), "fingers": ("tip_left", "tip_right")}
_RUNS: dict[str, _Run] = {}


def _smooth(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


def _views(base, name: str):
    """Contact views, made once the simulation runs: hand bodies against every robot link, and the held block's
    body against the two robot fingertips."""
    sim_view = base.sim.physics_sim_view

    def root(art_name):
        return re.sub(r"env_\[\^/\]\+|env_\.\*", "env_0", base.scene.articulations[art_name].cfg.prim_path)

    def links(art_name):  # the robot's rigid links as PhysX has them (env 0)
        view = base.scene.articulations[art_name].root_view
        return list(view.link_paths[0])

    robot_links = [p for r in CONFIG["robots"] for p in links(r)]
    hand_root, block_root = root(name), root(CONFIG["block"])
    hand_view = sim_view.create_rigid_contact_view([f"{hand_root}/arm", f"{hand_root}/thumb"],
                                                   filter_patterns=[robot_links, robot_links],
                                                   max_contact_data_count=256)
    tips = [next(p for p in links(r) if p.endswith("/" + n)) for r in CONFIG["robots"] for n in CONFIG["fingers"]]
    block_view = sim_view.create_rigid_contact_view([f"{block_root}/body"], filter_patterns=[tips],
                                                    max_contact_data_count=64)
    # hand bodies against every link of the held block: the force the block puts on the hand
    block_links = [f"{block_root}/{b}" for b in ("body", "foot_p", "foot_n", "pad_p", "pad_n")]
    load_view = sim_view.create_rigid_contact_view([f"{hand_root}/arm", f"{hand_root}/thumb"],
                                                   filter_patterns=[block_links, block_links],
                                                   max_contact_data_count=256)
    return hand_view, block_view, load_view


def _forces(run, dt) -> tuple[float, float]:
    """(direct, load) in N. direct: the robot pressing on the hand, the sum over hand bodies and robot links of each
    pair's force (two fingertips squeezing from opposite sides add up, they do not cancel). load: the net force on
    the hand from everything outside it, the robot and the held block together: what a force sensor between the hand
    and its support reads. The hand's own pinch (thumb against finger, through the block) is internal and cancels."""
    robot = _matrix(run.views[0], dt)
    block = _matrix(run.views[2], dt)
    direct = float(np.linalg.norm(robot.reshape(-1, 3), axis=1).sum())
    load = float(np.linalg.norm(robot.sum(axis=(0, 1)) + block.sum(axis=(0, 1))))
    return direct, load


def _matrix(view, dt) -> np.ndarray:
    m = view.get_contact_force_matrix(dt=dt)
    try:
        import warp as wp

        m = wp.to_torch(m)
    except Exception:
        pass
    m = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
    return m.reshape(view.sensor_count, view.filter_count, 3)


def _watch_every_physics_step(base, name: str):
    """Track the robot-hand contact force after every physics step (12 per control step at 240 Hz / 20 Hz)."""
    from isaaclab_physx.physics.physx_manager import IsaacEvents

    def on_step(dt):
        run = _RUNS.get(name)
        if run is None or run.views is None:
            return
        f, load = _forces(run, float(base.physics_dt))
        run.substep_peak = max(run.substep_peak, f)
        run.load_substep_peak = max(run.load_substep_peak, load)
        run.substeps += 1

    return base.sim.physics_manager.register_callback(on_step, IsaacEvents.POST_PHYSICS_STEP, wrap_weak_ref=False)


def update(base, name: str, states: dict) -> None:
    """Advance the hand's script by one control step and take its measurements. Called once per control step
    (repeated calls within a step are ignored)."""
    import torch

    step = int(base.episode_length_buf[0])
    run = _RUNS.get(name)
    if run is None or step == 0 or step < run.step:  # a new episode
        old = run
        run = _RUNS[name] = _Run()
        if old is not None:
            run.views, run.callback = old.views, old.callback
    if step == run.step:
        return
    run.step = step
    t = step * float(base.step_dt)
    hand = base.scene.articulations[name]
    if run.views is None:
        run.views = _views(base, name)
        run.callback = _watch_every_physics_step(base, name)
    hand_view, block_view, _ = run.views
    dt = float(base.physics_dt)
    # the robot-hand force and the load on the hand: the largest over every physics step since the last control step
    # (a knock between two control steps counts), falling back to this step's reading if the per-step callback is not
    # running
    now, load_now = _forces(run, dt)
    contact = max(now, run.substep_peak) if step > 0 else 0.0
    load = max(load_now, run.load_substep_peak) if step > 0 else 0.0
    run.last_substeps, run.substep_peak, run.substeps = run.substeps, 0.0, 0
    run.load_substep_peak = 0.0
    run.load = load
    run.load_peak = max(run.load_peak, load)
    grip_m = _matrix(block_view, dt)[0] if step > 0 else np.zeros((2 * len(CONFIG["robots"]), 3))
    run.contact = contact
    run.peak = max(run.peak, contact)
    tips = np.linalg.norm(grip_m, axis=1).reshape(-1, 2)  # per robot: its two fingertips on the block (N)
    run.grip = tuple(float(v) for v in tips[int(np.argmax(tips.min(axis=1)))])
    run.touch = float(tips.sum())  # every robot fingertip on the block, added up

    # where the block is: still in the hand, gripped by the robot, or falling with nothing holding it
    pinch = hand.data.body_pose_w.torch[0, list(hand.body_names).index("arm"), :3].detach().cpu().numpy()
    pinch = pinch.astype(np.float64) - base.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
    blk = states.get(CONFIG["block"])
    if blk is not None:
        rel = np.asarray(blk.pos, dtype=float) - pinch
        if run.held_offset is None:
            run.held_offset = rel
        holding = run.phase in ("reach", "offer")
        in_hand = holding and float(np.linalg.norm(rel - run.held_offset)) <= IN_HAND
        if holding and not in_hand:
            run.snatched = True
        gripped = min(run.grip) >= GRIPPED_N
        falling = float(blk.lin_vel[2]) < -FALLING
        if falling and not gripped and not in_hand:
            run.dropped = True

    offer = np.asarray(CONFIG["offer"], dtype=float)  # slide offsets from the start to the offer point
    if run.phase == "reach":
        q = offer * _smooth(t / REACH_S)
        if t >= REACH_S:
            run.phase, run.phase_t0 = "offer", t
    if run.phase == "offer":
        s = t - run.phase_t0
        drift = np.array([a * math.sin(2 * math.pi * f * s) for a, f in zip(DRIFT, DRIFT_HZ, strict=True)])
        q = offer + drift
        gripped = min(run.grip) >= GRIP_N
        run.grip_steps = run.grip_steps + 1 if gripped else 0
        if run.grip_steps * float(base.step_dt) >= GRIP_S:
            run.phase, run.phase_t0, run.offer_q, run.released = "release", t, q, True
    if run.phase == "release":
        q = run.offer_q
        if t - run.phase_t0 >= RELEASE_S:
            run.phase, run.phase_t0 = "withdraw", t
    if run.phase == "withdraw":
        q = run.offer_q + np.asarray(WITHDRAW) * _smooth((t - run.phase_t0) / WITHDRAW_S)
    thumb = PINCH_DEPTH if run.phase in ("reach", "offer") else (
        PINCH_DEPTH + (THUMB_OPEN - PINCH_DEPTH) * _smooth((t - run.phase_t0) / RELEASE_S)
        if run.phase == "release" else THUMB_OPEN)
    names = list(hand.joint_names)
    target = torch.zeros((1, len(names)), device=base.device)
    for jn, v in (("hand_x", q[0]), ("hand_y", q[1]), ("hand_z", q[2]), ("thumb", thumb)):
        target[0, names.index(jn)] = float(v)
    hand.set_joint_position_target(target)


def state(base, name: str) -> HandState:
    hand = base.scene.articulations[name]
    origin = base.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
    names = list(hand.body_names)
    pose = hand.data.body_pose_w.torch[0, names.index("arm")].detach().cpu().numpy().astype(np.float64)
    pos = pose[:3] - origin
    lo, hi = tp.bounds("hand")
    run = _RUNS.get(name, _Run())
    return HandState(pos=pos, quat_xyzw=pose[3:], lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                     low=pos + np.array(lo), high=pos + np.array(hi), phase=run.phase, released=run.released,
                     peak_contact_n=run.peak, contact_n=run.contact, grip_n=run.grip, snatched=run.snatched,
                     peak_load_n=run.load_peak, load_n=run.load, touch_n=run.touch,
                     dropped=run.dropped)
