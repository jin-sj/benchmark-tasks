"""Scripted two-arm demo for the conveyor-roller task. It uses only the task's known starting layout and the robot's own
joint readings (the measured gripper poses), never the simulator's object poses.

    python demo.py                    # replace the worn roller, worn one in the reject tray: 1.0
    python demo.py --variant one_end  # the new roller's left end in its hole, its right end left lying on the rail: 0.5
    python demo.py --variant removed  # the worn roller out and in the reject tray, the new one not fitted: 0.25
    python demo.py --variant hold     # no arm moves: 0
    python demo.py --device cpu       # CPU physics, for quick checks

Taking the worn roller out: the right hand, pointing down, fingers closing along the conveyor, takes hold of the worn
roller's tube in the middle (between its neighbours) and slides it 6.5 mm left, against the left rail. The left hand,
leaning out over the left rail, pinches the axle's left end where it sticks out of the rail and pulls it out until the
axle stops (16 mm): the axle's right end is drawn in clear of the right rail. Both hands then turn together about the
tube's left end, swinging the roller's right end up; the left hand lets go, and the right hand draws the axle's left
end out of its hole, lifts the roller out, turns it along the trays and lays it in the reject tray.

Fitting the new one: the right hand picks the new roller out of its tray by the middle, turns it across the conveyor
and sets it down on the two rails' tops (its axle ends on them: the hex settles flat, as the holes are), lets go and
takes hold of it again (now at a known place in the hand). It lifts it, slants it (right end high), brings the axle's
left end down inside the left rail and feels for the hole: the end pressed lightly on the web and swept down across
the hole's height, at a few offsets along the conveyor, until it slips in; then slides it in. It lowers the roller's
right end onto the right rail's top, lets go and takes a fresh hold. The left hand pulls the axle out at the left
(the right end comes in off the rail), both hands swing the roller level, the left hand takes a fresh pinch and pulls
again, and eases off: the spring presses the axle's right end on the right rail's web, and the right hand sweeps the
roller's end slowly along the conveyor, at a few heights, until the end pops out into the hole.

Writes ``result.json``, ``steps.jsonl`` and (unless ``--no-video``) one video per camera to ``--out``.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TASK = HERE / "task.py"
INSTRUCTION = "Replace the worn roller with the new one, and put the worn roller in the reject tray."
VARIANTS = ("full", "hold", "one_end", "removed")
CAMERAS = ("top_cam", "left_cam", "right_cam")
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ
MAX_STEPS = 4500
MM = 1e-3

TUBE_OPEN = 0.64  # gripper opening (0..1 of 94 mm) round the tube: 60 mm
TUBE_HOLD = 0.15  # closing on the 48 mm tube: the fingers stop on it, squeezing at the 50 N cap
STUB_OPEN = 0.36  # round the 11 mm hex axle end: 34 mm
STUB_HOLD = 0.0
TCP_ABOVE_TUBE = 6 * MM  # the TCP this far above the tube's axis when holding it
TCP_ABOVE_STUB = 4 * MM  # ... and above the axle's axis when pinching it
PINCH_OUT = 13.5 * MM  # pinch the axle end with the fingers' middle this far out from the rail's outer face
SLIDE_TO_RAIL = 6.5 * MM  # the tube's end to the left rail's web
PULL = 20.0 * MM  # pull the axle end out (command; it stops at its 16 mm travel: 12.7 mm clears the right rail)
LIFT_TILT = 8.0  # degrees: swing the right end up this far, pivoting on the left hole
LIFT_OFF = 15 * MM  # lift the new roller this far off the rails after taking hold of it again
INSERT_TILT = 5.0  # degrees: the right end this far up while the left axle end goes into its hole
SEARCH_DX = (-9.0, -8.0, -10.0, -7.0, 0.0, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0)  # mm: the axle end's sweeps over the hole, offset this much along the conveyor
SEARCH_Z = 5 * MM  # each sweep runs from this far above the hole's centre (as the hand puts it) to as far below
SEARCH_S = 4.0  # seconds a sweep takes
PRESS_IN = 3 * MM  # the hand aims the axle end this far past the web's face (the arm gives: a light push)
IN_HOLE = 1.5 * MM  # the axle end (as the hand puts it) this far past the web's face: it is in the hole
SWEEP_B = ((-12 * MM, 2.5), (6 * MM, 5.0), (0.0, 1.5))  # the hand's sweep along the conveyor: (offset, seconds) ...
SWEEP_BZ = (3.0 * MM, 3.5 * MM, 2.5 * MM, 2.0 * MM, 1.0 * MM)  # ... at these heights above its level height, in turn
PINCH_TILT = 60.0  # degrees: the left gripper, pinching an axle end, leans out over the rail (its fingers pointing in and down)
AFTER_SUCCESS_S = 3.0


def smooth(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Arm:
    """One arm: its target TCP pose and gripper, and a correction (integrated from the measured TCP error) added to the
    commanded position, turned into joint targets by IK (warm-started)."""

    def __init__(self, side: str, base, q0, ik, grip_rotation):
        self.side, self.base, self.ik = side, np.asarray(base, dtype=float), ik
        self.q = np.array(q0, dtype=float)
        self.target, self.rot = ik.tcp_pose(self.q, self.base)
        self.corr = np.zeros(3)
        self.rcorr = np.zeros(3)  # the same for the orientation (a rotation vector, world frame)
        self.track = False
        self.rtrack = False  # also correct the orientation
        self.mask = np.ones(3)  # which axes the correction acts on
        self.tool = None  # a point held in the hand (TCP frame): when set, the correction brings it, not the TCP, to
        #                   where the target puts it
        self.gain, self.limit = 0.12, 0.08
        self.grip = 1.0
        self.down = grip_rotation(0.0)

    def command(self) -> np.ndarray:
        return self.target + self.corr

    def command_rot(self) -> np.ndarray:
        from yam_arena.urdf import axis_angle_matrix

        a = float(np.linalg.norm(self.rcorr))
        return self.rot if a < 1e-9 else axis_angle_matrix(self.rcorr / a, a) @ self.rot

    def update(self, measured, measured_rot) -> None:
        if self.track:
            err = self.target - measured
            if self.tool is not None:
                err = err + (self.rot - measured_rot) @ self.tool
            self.corr = np.clip(self.corr + self.gain * self.mask * err, -self.limit, self.limit)
        if self.track and self.rtrack:
            err = self.ik._rot_error(self.rot, measured_rot)
            self.rcorr = np.clip(self.rcorr + self.gain * err, -0.25, 0.25)

    def reset_corr(self) -> None:
        self.corr[:] = 0
        self.rcorr[:] = 0

    def action(self) -> np.ndarray:
        q, pe, re = self.ik.solve(self.command(), self.ik.matrix_to_quat_xyzw(self.command_rot()), self.q,
                                  base_pos=self.base, iters=60)
        self.q, self.ik_err = q, (pe, re)
        return np.r_[q, self.grip]


def make_embodiment(video: bool, device: str = "cuda:0"):
    from yam_arena.inspect_embodiment import YamArenaEmbodiment

    size = (360, 640) if video else (64, 64)
    return YamArenaEmbodiment(task=str(TASK), control_hz=CONTROL_HZ, object_jitter=0.0, cam_height=size[0],
                              cam_width=size[1], device=device)


def set_rendering(emb, on: bool) -> None:
    """Turn the RTX render that follows every control step on or off (the physics and scoring are unaffected)."""
    emb._env.unwrapped.cfg.sim.render_interval = emb._env.unwrapped.cfg.decimation if on else 10**9


def run(variant: str, out: Path, video: bool, max_steps: int, emb=None) -> dict:
    from inspect_robots import Scene
    from inspect_robots.types import Action
    from inspect_robots_yam.packing import pack, split

    from yam_arena import ik, station
    from yam_arena.demo import grip_rotation

    emb = emb or make_embodiment(video)
    obs = emb.reset(Scene(id=f"conveyor-roller-{variant}", instruction=INSTRUCTION), seed=0)
    set_rendering(emb, video)
    task = emb.task.module
    rp = task.rp
    from yam_arena.scene import object_states  # patched by the task: its own parts

    left_q, right_q = split(np.asarray(obs.state["joint_pos"]))
    arms = {
        "left": Arm("left", station.BASES["left"], left_q[:6], ik, grip_rotation),
        "right": Arm("right", station.BASES["right"], right_q[:6], ik, grip_rotation),
    }
    arms["left"].limit = 0.03  # the left hand pulls on the spring: its correction is kept moderate
    world = {"obs": obs, "step": 0}
    report = {}
    phase = {"left": "", "right": ""}

    def measured_tcp(side: str):
        left, right = split(np.asarray(world["obs"].state["joint_pos"]))
        q = (left if side == "left" else right)[:6]
        return ik.tcp_pose(q, arms[side].base)

    def move(arm: Arm, pos, duration: float, grip=None, rot=None, label=""):
        phase[arm.side] = label
        p0, r0, g0 = arm.target.copy(), arm.rot.copy(), arm.grip
        p1 = np.asarray(pos, dtype=float)
        r1 = r0 if rot is None else rot
        g1 = g0 if grip is None else grip
        n = max(1, round(duration / DT))
        for i in range(1, n + 1):
            s = smooth(i / n)
            arm.target = p0 + s * (p1 - p0)
            arm.rot = ik.slerp_matrix(r0, r1, s)
            arm.grip = g0 + s * (g1 - g0)
            yield

    def path(arm: Arm, fn, duration: float, label=""):
        """Follow ``fn(s) -> (pos, rot)`` for s from 0 to 1 (smoothstep in time)."""
        phase[arm.side] = label
        n = max(1, round(duration / DT))
        for i in range(1, n + 1):
            arm.target, arm.rot = fn(smooth(i / n))
            yield

    def settle(arm: Arm, label, tol=0.6 * MM, hold=4, max_s=3.0):
        """Wait until the measured TCP has been within ``tol`` of the target for ``hold`` steps (or ``max_s``)."""
        phase[arm.side] = label
        ok = 0
        for _ in range(round(max_s / DT)):
            err = arm.target - measured_tcp(arm.side)[0]
            ok = ok + 1 if float(np.linalg.norm(err)) <= tol else 0
            if ok >= hold:
                return
            yield

    def wait(seconds: float, arm: Arm | None = None, label=""):
        if arm is not None:
            phase[arm.side] = label
        for _ in range(round(seconds / DT)):
            yield

    def together(*gens):
        live = list(gens)
        while live:
            for g in list(live):
                try:
                    next(g)
                except StopIteration:
                    live.remove(g)
            yield

    def idle():
        while True:
            yield

    L, R = arms["left"], arms["right"]
    down = R.down
    across = rot_z(math.pi / 2) @ down  # pointing down, fingers closing along x (along the conveyor)
    along = down  # pointing down, fingers closing along y (across the conveyor)
    zh = task.HOLE[1] - task.SEATED_DROP  # a seated axle's centre height
    x0 = task.SLOT_X
    pivot_y = rp.INNER_Y + rp.WEB_T / 2  # the left hole's middle (y)
    half = rp.AXLE_LEN / 2
    up_tube = np.array([0.0, 0.0, TCP_ABOVE_TUBE])
    park = np.array([0.22, 0.33, task.TOP + 0.20])  # the left hand's waiting place, clear of the conveyor

    def tilted(theta_deg: float, centre) -> tuple[np.ndarray, np.ndarray]:
        """The right hand's (TCP, rotation) holding a roller whose middle is ``centre``, right end raised by theta."""
        r = rot_x(-math.radians(theta_deg))
        return np.asarray(centre, dtype=float) + r @ up_tube, r @ across

    def axis(theta_deg: float) -> np.ndarray:
        t = math.radians(theta_deg)
        return np.array([0.0, math.cos(t), -math.sin(t)])  # toward the roller's left end

    def pinch_point(tip_y: float, z: float) -> np.ndarray:
        """The left TCP pinching the left axle end (the fingers' middle ``PINCH_OUT`` out of the rail)."""
        return np.array([x0, rp.OUTER_Y + PINCH_OUT, z + TCP_ABOVE_STUB])

    def remove_worn():
        # right hand on the worn roller's middle
        c0 = np.array([x0, 0.0, zh])
        yield from move(R, c0 + up_tube + (0, 0, 0.10), 2.5, grip=TUBE_OPEN, rot=across, label="over the worn roller")
        R.track = True
        yield from settle(R, "line up over it")
        yield from move(R, c0 + up_tube, 1.5, label="down round its tube")
        yield from settle(R, "line up on the tube")
        yield from move(R, R.target, 0.8, grip=TUBE_HOLD, label="take hold of the tube")
        c1 = c0 + (0, SLIDE_TO_RAIL, 0)
        yield from move(R, c1 + up_tube, 1.0, label="slide it against the left rail")
        report["worn_held"] = world["step"]

    def left_pull(z_stub: float, label: str):
        p = pinch_point(0.0, z_stub)
        lean = rot_x(-math.radians(PINCH_TILT)) @ across
        back = -lean[:, 2]  # back along the gripper's approach axis
        yield from move(L, p + 0.06 * back, 2.5, grip=STUB_OPEN, rot=lean, label=f"over the axle end ({label})")
        L.track = True
        yield from settle(L, "line up over the axle end")
        yield from move(L, p, 1.5, label="down round the axle end")
        yield from settle(L, "line up on the axle end")
        yield from move(L, p, 0.8, grip=STUB_HOLD, label="pinch the axle end")
        yield from move(L, p + (0, PULL, 0), 1.5, label="pull the axle end out")
        yield from settle(L, "hold it out", tol=1.0 * MM, max_s=2.0)

    def swing(d_theta: float, duration: float, label: str, pivot=None):
        """Both hands turn together (the left hand pinching the axle end, the right holding the tube) about the top
        edge of the tube's left end, against the left rail: the roller's right end rises by ``d_theta`` degrees (falls
        if negative)."""
        pv = np.array([x0, rp.INNER_Y - 0.5 * MM, zh + rp.TUBE_R]) if pivot is None else pivot
        starts = {a.side: (a.target.copy(), a.rot.copy()) for a in (L, R)}

        def mk(a):
            p0, r0 = starts[a.side]

            def fn(s_):
                rr = rot_x(-math.radians(d_theta * s_))
                return pv + rr @ (p0 - pv), rr @ r0
            return fn

        yield from together(path(R, mk(R), duration, label), path(L, mk(L), duration, label))

    def lift_worn():
        yield from swing(LIFT_TILT, 2.5, "swing its right end up")
        report["worn_swung"] = world["step"]

    def take_worn_out():
        yield from move(L, L.target, 0.6, grip=STUB_OPEN, label="let the axle end go")
        L.track = False
        L.reset_corr()
        yield from move(L, L.target + 0.05 * (-L.rot[:, 2]), 1.0, label="back off")
        yield from move(L, park, 1.5, rot=across, label="out of the way")
        c = R.target - tilted(LIFT_TILT, np.zeros(3))[0]  # the roller's middle now
        yield from move(R, tilted(LIFT_TILT, c + (0, -0.035, 0.01))[0], 1.5, label="draw its left end out of the hole")
        yield from move(R, tilted(0.0, c + (0, -0.035, 0.13))[0], 2.0, rot=across, label="lift it clear")
        tray = np.array([task.REJECT_ORIGIN[0], task.REJECT_ORIGIN[1], task.TOP + rp.TRAY_FLOOR + rp.TUBE_R])
        yield from move(R, tray + up_tube + (0, 0, 0.12), 3.0, rot=along, label="over the reject tray")
        yield from move(R, tray + up_tube + (0, 0, 0.004), 2.0, label="lay it in the tray")
        yield from move(R, R.target, 0.8, grip=TUBE_OPEN, label="let go")
        yield from move(R, R.target + (0, 0, 0.10), 1.2, label="up")
        report["worn_in_tray"] = world["step"]

    def fit_new(one_end: bool):
        tray = np.array([task.TRAY_ORIGIN[0], task.TRAY_ORIGIN[1], task.TOP + rp.TRAY_FLOOR + rp.TUBE_R + 0.5 * MM])
        yield from move(R, tray + up_tube + (0, 0, 0.08), 2.0, grip=TUBE_OPEN, rot=along, label="over the new roller")
        yield from settle(R, "line up over it")
        yield from move(R, tray + up_tube, 1.5, label="down round its tube")
        yield from settle(R, "line up on the tube")
        yield from move(R, R.target, 0.8, grip=TUBE_HOLD, label="take hold of it")
        report["new_held"] = world["step"]
        yield from move(R, tray + up_tube + (0, 0, 0.12), 1.5, label="lift it out")
        # across the conveyor, set down on the two rails' tops (its axle ends on them), and taken hold of again
        rest = np.array([x0, 0.0, task.TOP + rp.FRAME_H + rp.AXLE_AF / 2])  # its middle, resting on the rails
        yield from move(R, rest + up_tube + (0, 0, 0.10), 3.0, rot=across, label="turn it across the conveyor")
        yield from settle(R, "over the slot", tol=1.0 * MM)
        yield from move(R, rest + up_tube + (0, 0, 0.002), 1.5, label="set it on the rails")
        yield from move(R, R.target, 0.8, grip=TUBE_OPEN, label="let go")
        yield from move(R, R.target + (0, 0, 0.004), 0.5, label="let it settle")
        yield from move(R, rest + up_tube, 0.8, label="down round its tube")
        yield from settle(R, "line up on the tube")
        yield from move(R, R.target, 0.8, grip=TUBE_HOLD, label="take hold of it again")
        report["new_on_rails"] = world["step"]
        ca = rest + (0, 0, LIFT_OFF)
        yield from move(R, ca + up_tube, 1.0, label="lift it off the rails")
        yield from settle(R, "hold it", tol=0.6 * MM)
        # where the axle's left end is in the hand (the hand sags a few degrees under the roller): from the right
        # hand's measured pose; from here its correction steers that end, not the hand
        tip_in_hand = R.rot.T @ (ca + (0, half, 0) - R.target)
        tip_b_in_hand = R.rot.T @ (ca - (0, half, 0) - R.target)
        # the left end into its hole, the right end high: from here the right hand's correction steers the axle's
        # left end (where it is in the hand, from the measured pose), not the hand
        R.tool = tip_in_hand
        th = INSERT_TILT
        hole = np.array([x0, rp.INNER_Y, task.HOLE[1]])  # the left hole's mouth (inner face), its centre

        def tcp_for_tip(tip, theta=th):
            return tilted(theta, np.asarray(tip, dtype=float) - half * axis(theta))

        def tip_now():
            pm, rm = measured_tcp("right")
            return pm + rm @ R.tool

        p0, r0 = tcp_for_tip(hole + (0, -4 * MM, 3 * MM))
        yield from move(R, p0 + (0, 0, 0.03), 2.5, rot=r0, label="slant it over the slot")
        yield from settle(R, "line up", tol=0.5 * MM)
        yield from move(R, p0, 1.5, label="left axle end down inside the rail")
        yield from settle(R, "line up by the hole", tol=0.4 * MM)
        # feel for the hole: the axle end pressed lightly on the rail's web (the hand's push along y left to the arm's
        # give), swept down across where the hole is, at a few offsets along the conveyor, until it goes in
        R.mask = np.array([1.0, 0.0, 1.0])
        found = False
        for dx in SEARCH_DX:
            top = hole + (dx * MM, PRESS_IN, SEARCH_Z)
            yield from move(R, tcp_for_tip(top)[0], 0.8, label="press the axle end on the web")
            n = round(SEARCH_S / DT)
            for i in range(1, n + 1):
                R.target = tcp_for_tip(top - (0, 0, 2 * SEARCH_Z * i / n))[0]
                phase["right"] = "sweep it down over the hole"
                yield
                if tip_now()[1] > rp.INNER_Y + IN_HOLE:
                    found = True
                    break
            if found:
                break
            yield from move(R, tcp_for_tip(hole + (dx * MM, -2 * MM, SEARCH_Z))[0], 0.6, label="back off the web")
        report["found_hole"] = world["step"] if found else None
        here = tip_now()  # (the hand's push along y stays left to the arm's give from here on)
        end_tip = np.array([here[0], rp.INNER_Y + rp.STUB + 1 * MM, here[2]])
        yield from move(R, tcp_for_tip(end_tip)[0], 2.0, label="slide the axle end in")
        yield from settle(R, "slide it home", tol=0.8 * MM, max_s=2.0)
        report["left_in"] = world["step"]
        # let the roller's right end down onto the right rail's top (its left end stays in its hole), and take a
        # fresh hold of the tube, where it now lies is known: the hand slips on the tube while the end is fed in
        yield from move(R, R.target + (0, 0, -0.010), 1.0, label="lower its right end toward the rail")
        yield from move(R, R.target, 0.8, grip=TUBE_OPEN, label="rest its right end on the rail")
        R.track = False
        R.mask = np.ones(3)
        R.tool = None
        R.reset_corr()
        yield from move(R, R.target + (0, 0, 0.04), 1.0, label="up off the tube")
        report["resting"] = world["step"]
        if one_end:
            yield from move(R, R.target + (0, -0.02, 0.06), 1.2, label="up")
            return
        z_a = task.HOLE[1] - task.SEATED_DROP  # the axle in the left hole ...
        z_b = task.TOP + rp.FRAME_H + rp.AXLE_AF / 2  # ... and lying on the right rail's top
        rest_th = math.degrees(math.atan2(z_b - z_a, 2 * pivot_y))
        yc = rp.INNER_Y - rp.TUBE_LEN / 2  # the tube's left end against the left rail
        c_rest = np.array([x0, yc, z_a + (z_b - z_a) * (pivot_y - yc) / (2 * pivot_y)])
        p_rest, r_rest = tilted(rest_th, c_rest)
        yield from move(R, p_rest + (0, 0, 0.04), 1.2, rot=r_rest, label="over the tube")
        R.track = True
        yield from settle(R, "line up over the tube")
        yield from move(R, p_rest, 1.2, label="down round the tube")
        yield from settle(R, "line up on the tube")
        yield from move(R, R.target, 0.8, grip=TUBE_HOLD, label="take hold of it again")
        R.mask = np.array([1.0, 0.0, 1.0])  # along the conveyor's width the hand is left to the arm's give
        # pull the axle out at the left (its right end comes off the rail), swing the roller level, ease the axle back
        z_stub = z_a + (rp.OUTER_Y + PINCH_OUT - pivot_y) * -math.sin(math.radians(rest_th))
        yield from left_pull(z_stub, "outside the left rail")
        yield from swing(-rest_th, 2.0, "swing it level", pivot=np.array([x0, pivot_y, z_a]))
        yield from move(L, L.target, 0.5, grip=STUB_OPEN, label="let the axle end go")
        yield from move(L, L.target + 0.04 * (-L.rot[:, 2]), 0.6, label="back off")
        L.reset_corr()
        yield from left_pull(z_a, "again")
        level = tilted(0.0, np.array([x0, yc, z_a]))[0]
        yield from move(R, level, 0.8, label="right end level with its hole")
        yield from settle(R, "hold it there", tol=0.4 * MM, max_s=2.0)
        report["level"] = world["step"]
        yield from move(L, L.target - (0, PULL + 2 * MM, 0), 2.5, label="ease the axle back")
        yield from move(L, L.target, 0.6, grip=STUB_OPEN, label="let go")
        L.track = False
        L.reset_corr()
        # the spring now presses the axle's right end on the web: the hand sweeps it slowly along the conveyor across
        # the hole (the tube's middle is half way along the axle, so the end moves twice as far) until it pops in
        for dz in SWEEP_BZ:
            for dx, t in SWEEP_B:
                yield from move(R, level + (dx, 0, dz), t, label="feel for the right hole")
        yield from move(R, level, 0.6, label="right end in its hole")
        yield from move(L, L.target + 0.05 * (-L.rot[:, 2]), 1.0, label="back off")
        yield from move(L, park, 1.5, rot=across, label="out of the way")
        yield from move(R, R.target, 0.8, grip=TUBE_OPEN, label="let go")
        R.track = False
        R.mask = np.ones(3)
        R.tool = None
        R.reset_corr()
        yield from move(R, R.target + (0, 0, 0.10), 1.2, label="up")
        report["fitted"] = world["step"]

    def script():
        if variant == "hold":
            phase["left"] = phase["right"] = "idle"
            yield from idle()
        yield from remove_worn()
        yield from left_pull(zh, "the worn roller")
        report["pulled"] = world["step"]
        yield from lift_worn()
        yield from take_worn_out()
        if variant != "removed":
            yield from fit_new(one_end=variant == "one_end")
        phase["left"] = phase["right"] = "done"
        report["script_done_step"] = world["step"]
        yield from idle()

    driver = script()
    frames = {c: [] for c in CAMERAS} if video else None
    if video:
        for c in CAMERAS:
            frames[c].append(obs.images[c])
    log = []
    result = None
    success_step = None
    t0 = time.time()
    for step in range(1, max_steps + 1):
        world["step"] = step
        next(driver)
        a = pack(L.action(), R.action())
        res = emb.step(Action(data=a))
        world["obs"] = res.observation
        for arm in (L, R):
            arm.update(*measured_tcp(arm.side))
        st = object_states(emb._env, emb.task.names)  # logged only: the script never reads object poses
        row = {"step": step, "left": phase["left"], "right": phase["right"], "score": res.info.get("score"),
               "success": res.info.get("success")}
        for n in ("worn_roller", "new_roller"):
            s = st[n]
            row[n] = {"pos": [round(float(v), 4) for v in s.pos], "tl": [round(float(v), 4) for v in s.tip_left],
                      "tr": [round(float(v), 4) for v in s.tip_right], "slide_mm": round(s.slide * 1000, 2),
                      "hex": round(s.hex_deg, 1)}
        row["ends"] = task.ends_seated(st)
        for arm in (L, R):
            k = arm.side[0]
            row[f"{k}_tgt"] = [round(float(v), 4) for v in arm.target]
            row[f"{k}_meas"] = [round(float(v), 4) for v in measured_tcp(arm.side)[0]]
            row[f"{k}_corr"] = [round(float(v) * 1000, 1) for v in arm.corr]
            row[f"{k}_rerr_deg"] = round(math.degrees(float(np.linalg.norm(ik._rot_error(arm.rot, measured_tcp(arm.side)[1])))), 2)
        log.append(row)
        if video:
            for c in CAMERAS:
                frames[c].append(res.observation.images[c])
        done_script = "script_done_step" in report and step - report["script_done_step"] >= round(3.0 / DT)
        if res.terminated and success_step is None:
            success_step, result = step, res
            report["success_step"] = step
        if success_step is not None:
            if step - success_step >= round(AFTER_SUCCESS_S / DT):
                break
            continue
        result = res
        if res.terminated or done_script:
            break
    success = bool(result.terminated and result.termination_reason == "success")
    st = object_states(emb._env, emb.task.names)
    summary = {
        "variant": variant,
        "task_score": 1.0 if success else float(result.info.get("score", 0.0)),
        "terminated": bool(result.terminated),
        "termination_reason": result.termination_reason,
        "steps": len(log),
        "run_limit_steps": MAX_STEPS,
        "worn_in_tray": task.in_reject_tray(st),
        "new_ends_seated": task.ends_seated(st),
        "report": report,
        "wall_s": time.time() - t0,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(summary, indent=2, default=float))
    with open(out / "steps.jsonl", "w") as f:
        for row in log:
            f.write(json.dumps(row, default=float) + "\n")
    if video:
        write_videos(frames, out, variant)
        import cv2

        cv2.imwrite(str(out / "setup.png"), np.ascontiguousarray(frames["top_cam"][0][..., ::-1]))
    return summary


def write_videos(frames: dict, out: Path, variant: str) -> None:
    import cv2

    for cam, imgs in frames.items():
        if not imgs:
            continue
        h, w = imgs[0].shape[:2]
        raw = out / f"_{cam}.avi"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"MJPG"), CONTROL_HZ, (w, h))
        for im in imgs:
            writer.write(np.ascontiguousarray(im[..., ::-1]))
        writer.release()
        name = out / (f"demo_{cam}.mp4" if variant == "full" else f"demo_{variant}_{cam}.mp4")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", str(name)], check=True)
            raw.unlink()
        else:
            raw.rename(name.with_suffix(".avi"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=VARIANTS + ("all",), default="full",
                    help="'all' runs every variant in one simulator session (reset between them)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: runs/<variant>)")
    ap.add_argument("--no-video", action="store_true", help="no videos (and no rendering)")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--device", default="cuda:0",
                    help="physics device; 'cpu' is faster for this one-environment scene, but its results differ in "
                         "detail from the GPU solver trials run on: confirm on cuda:0")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override a demo constant for an experiment (a Python literal), e.g. --set PULL=0.017")
    args = ap.parse_args(argv)
    for item in args.set:
        name, value = item.split("=", 1)
        if name not in globals():
            ap.error(f"--set: no constant {name!r}")
        globals()[name] = ast.literal_eval(value)
    video = not args.no_video
    emb = make_embodiment(video, args.device)
    for variant in VARIANTS if args.variant == "all" else (args.variant,):
        out = args.out / variant if args.out and args.variant == "all" else args.out or HERE / "runs" / variant
        summary = run(variant, out, video, args.max_steps, emb)
        print("RESULT " + json.dumps({k: summary[k] for k in ("variant", "task_score", "steps", "termination_reason",
                                                               "worn_in_tray", "new_ends_seated", "wall_s")}))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
