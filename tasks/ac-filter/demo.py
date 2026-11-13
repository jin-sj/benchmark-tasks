"""Scripted two-arm demo for the air-conditioner filter task. It uses only the task's known starting layout and the
robot's own joint readings (the measured gripper positions), never the simulator's object poses.

    python demo.py                    # open the cover, swap the filters, close the cover: 1.0
    python demo.py --variant open     # the same, but the cover is left open: 0.5
    python demo.py --variant notfit   # the clean filter is left half fitted (its top behind the lip): 0.25
    python demo.py --variant outonly  # only the dusty filter is taken out: 0.25
    python demo.py --variant hold     # no arm moves: 0
    python demo.py --device cpu       # CPU physics, ~3x faster, for quick checks

The left arm, its gripper closed and pointing forward, lifts the cover by its finger pull until it springs open.
Both arms then pinch the dusty filter's two pull tabs (pointing forward, fingers closing up and down), lift it off its
lugs, swing its bottom out, lower its top past the lip, bring it out and lay it face up on the table near the robot.
They pinch the clean filter's tabs from above, turn it up and bring it to the unit tilted, push its top up behind the
lip, swing it upright over the lugs and set it down on the ledge. The left arm presses the cover down until it
springs shut.

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
INSTRUCTION = ("Replace the dusty filter with the clean one: open the cover, take the dusty filter out, fit the clean "
               "filter in its place, and close the cover.")
VARIANTS = ("full", "hold", "outonly", "notfit", "open")
CAMERAS = ("top_cam", "left_cam", "right_cam")
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ
MAX_STEPS = 3000
MM = 1e-3

PAD_TIP = 0.013  # pad tips beyond the TCP
OPEN = 0.35  # gripper opening (0..1 of 94 mm) around a pull tab
GRIP_Y = 0.15  # the tabs, left and right of a filter's middle
TAB_GRIP_X = -21 * MM  # the pinch's TCP on a tab, in the filter's frame: the pad tips 8 mm short of its root
DUSTY_DOWN_X = 0.15  # lay the dusty filter down with its tabs here (from the arm bases)
LIFT = 9 * MM  # lift the dusty filter this much (its bottom clears the 6 mm lugs) ...
LIFT_BACK = 0.0  # ... pushing it back (into the unit) this much as it goes up
TILT_DEG = 10.0  # ... swing its bottom out this far, about its top ...
DROP = 32 * MM  # ... and lower its top this much (clear of the lip)
FIT_TILT_DEG = 8.0  # bring the clean filter in tilted this much, bottom out ...
FIT_TOP_X = 13.5 * MM  # ... its top edge this far behind the unit's front (behind the lip), just under the lip
FIT_RAISE = 28 * MM  # then raise it this much (its top up behind the lip, its bottom above the lugs)
FIT_ABOVE = 10 * MM  # swing it upright this far above its place, then set it down ...
FIT_DEEPER = 1 * MM  # ... this much deeper than its place (well behind the lugs)
FINGER_SIDE = 6.5 * MM  # a closed gripper's fingers' outer face near their tips, from the TCP
PRESS = 3 * MM  # push into the cover this much
PUSH_UP_PRESS = 7 * MM  # pushing its bottom edge up from below, this much
OPEN_FIRST_DEG = 35.0  # swing the cover out by its pull this far, then push its bottom edge up to the stop
CLOSE_AT = 165 * MM  # to close it, press on its face this far from the hinge ...
CLOSE_MID_DEG = 40.0  # ... down to this angle, then push it shut with the fingertips ...
CLOSE_TIPS_AT = 120 * MM  # ... on its face this far from the hinge
COVER_Y = 0.25  # the left arm works the cover here (clear of the filter's tabs, at +-0.15)
AFTER_SUCCESS_S = 2.0
PINCH_GRIP = 0.0  # the gripper command when pinching a tab (0: shut)
CARRY_GAIN = 0.15  # carrying a filter, each arm's command corrects by this much of its measured error each step ...
CARRY_LIMIT = 25 * MM  # ... up to this far

def smooth(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


class Arm:
    """One arm's commanded TCP pose and gripper, turned into joint targets by IK (warm-started)."""

    def __init__(self, side: str, base, q0, ik, grip_rotation):
        self.side, self.base, self.ik = side, np.asarray(base, dtype=float), ik
        self.q = np.array(q0, dtype=float)
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.grip = 1.0
        self.down = grip_rotation(0.0)

    def action(self) -> np.ndarray:
        q, pe, re = self.ik.solve(self.pos, self.ik.matrix_to_quat_xyzw(self.rot), self.q, base_pos=self.base, iters=60)
        self.q, self.ik_err = q, (pe, re)
        return np.r_[q, self.grip]


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def make_embodiment(video: bool, device: str = "cuda:0"):
    from yam_arena.inspect_embodiment import YamArenaEmbodiment

    size = (360, 640) if video else (64, 64)
    return YamArenaEmbodiment(task=str(TASK), control_hz=CONTROL_HZ, object_jitter=0.0, cam_height=size[0],
                              cam_width=size[1], device=device)


def set_rendering(emb, on: bool) -> None:
    """Turn the RTX render that follows every control step on or off. Off, the cameras keep their last frame;
    the physics, the object states and the task's scoring are unaffected (the script reads object poses, not
    images)."""
    emb._env.unwrapped.cfg.sim.render_interval = emb._env.unwrapped.cfg.decimation if on else 10**9


def run(variant: str, out: Path, video: bool, max_steps: int, emb=None) -> dict:
    from inspect_robots import Scene
    from inspect_robots.types import Action
    from inspect_robots_yam.packing import pack, split

    from yam_arena import ik, station
    from yam_arena.demo import grip_rotation

    emb = emb or make_embodiment(video)
    obs = emb.reset(Scene(id=f"ac-filter-{variant}", instruction=INSTRUCTION), seed=0)
    set_rendering(emb, video)
    task = emb.task.module
    from yam_arena.scene import object_states  # patched by the task: its own parts

    left_q, right_q = split(np.asarray(obs.state["joint_pos"]))
    arms = {
        "left": Arm("left", station.BASES["left"], left_q[:6], ik, grip_rotation),
        "right": Arm("right", station.BASES["right"], right_q[:6], ik, grip_rotation),
    }
    world = {"states": object_states(emb._env, emb.task.names), "obs": obs, "step": 0}
    report = {}
    phase = {"left": "", "right": ""}

    def measured_tcp(side: str):
        left, right = split(np.asarray(world["obs"].state["joint_pos"]))
        q = (left if side == "left" else right)[:6]
        return ik.tcp_pose(q, arms[side].base)

    def move(arm: Arm, pos, duration: float, grip=None, rot=None, label=""):
        phase[arm.side] = label
        p0, r0, g0 = arm.pos.copy(), arm.rot.copy(), arm.grip
        p1 = np.asarray(pos, dtype=float)
        r1 = r0 if rot is None else rot
        g1 = g0 if grip is None else grip
        n = max(1, round(duration / DT))
        for i in range(1, n + 1):
            s = smooth(i / n)
            arm.pos = p0 + s * (p1 - p0)
            arm.rot = ik.slerp_matrix(r0, r1, s)
            arm.grip = g0 + s * (g1 - g0)
            yield

    def settle_until(arm, target, label, tol=0.8 * MM, hold=4, max_s=3.0, gain=0.15, limit=0.015):
        """Hold ``target``, correcting the command by the measured TCP error, until the measured TCP has been within
        ``tol`` of it for ``hold`` steps in a row (or ``max_s`` has passed)."""
        phase[arm.side] = label
        target = np.asarray(target, dtype=float)
        corr, ok = arm.pos - target, 0
        for _ in range(round(max_s / DT)):
            err = target - measured_tcp(arm.side)[0]
            ok = ok + 1 if float(np.linalg.norm(err)) <= tol else 0
            if ok >= hold:
                break
            corr = np.clip(corr + gain * err, -limit, limit)
            arm.pos = target + corr
            yield

    def together(*gens):
        """Run several arms' motions side by side, one step each per control step, until all are done."""
        live = list(gens)
        while live:
            for g in list(live):
                try:
                    next(g)
                except StopIteration:
                    live.remove(g)
            yield

    def both(make):
        """``make(arm, y)`` for both arms (y: the gripper's offset across), side by side."""
        return together(make(arms["left"], GRIP_Y), make(arms["right"], -GRIP_Y))

    def idle():
        while True:
            yield

    ap = task.ap
    F, T = task.FACE_X, task.TOP
    sides = {"left": GRIP_Y, "right": -GRIP_Y}
    down = arms["left"].down
    # the grips: down, the fingers closing along x (each arm turned the way that keeps its wrist in range), and the
    # same turned to point forward (toward the unit), the fingers closing up and down
    down_x = {"left": rot_z(-math.pi / 2) @ down, "right": rot_z(math.pi / 2) @ down}
    fwd = {s: rot_y(-math.pi / 2) @ down_x[s] for s in sides}
    bx, bz = ap.bend()
    p_top = np.array([bx, 0.0, ap.FILTER_H])  # a filter's top edge (front), in its frame
    p_tab = np.array([TAB_GRIP_X, 0.0, ap.TAB[2] / 2])  # the pinch on its tabs (TCP), in its frame (y: each tab)
    lie = math.radians(task.LIE_DEG)
    hinge = np.array([F + ap.HINGE[0], T + ap.HINGE[1]])  # (x, z)

    def filt(a, pivot_local, pivot_world):
        """A filter's pose (origin, rotation): turned ``a`` (rad) about +y, its point ``pivot_local`` at
        ``pivot_world``."""
        r = rot_y(a)
        return np.asarray(pivot_world, dtype=float) - r @ pivot_local, r

    seated = (np.asarray(task.DUSTY_ORIGIN, dtype=float), np.eye(3))
    lying_clean = (np.asarray(task.CLEAN_ORIGIN, dtype=float), rot_y(lie))
    lying_dusty = (np.asarray(task.CLEAN_ORIGIN, dtype=float) + np.array([DUSTY_DOWN_X - task.CLEAN_X, 0.0, 0.0]),
                   rot_y(lie))

    def grips(pose):
        """Each arm's TCP and rotation pinching the tabs of a filter at ``pose``."""
        o, r = pose
        return {s: (o + r @ (p_tab + np.array([0.0, y, 0.0])), r @ fwd[s]) for s, y in sides.items()}

    lead = {s: np.zeros(3) for s in sides}  # each arm's command lead (its load) while it holds a filter
    lead["cover"] = np.zeros(3)  # the left arm's while it works the cover

    def to_grasp(pose, label, approach):
        """Both arms: open, come to ``approach`` away from the tabs, line up, close in, line up, pinch."""
        g = grips(pose)

        def make(arm, y):
            p, r = g[arm.side]
            near = p + np.asarray(approach, dtype=float)
            yield from move(arm, near, 2.5, grip=OPEN, rot=r, label=f"{label}: line up")
            yield from settle_until(arm, near, f"{label}: line up")
            yield from move(arm, p + (arm.pos - near), 1.2, label=f"{label}: close in")
            yield from settle_until(arm, p, f"{label}: line up on the tabs", tol=0.8 * MM)
            yield from move(arm, arm.pos, 0.8, grip=PINCH_GRIP, label=f"{label}: pinch the tabs")
            lead[arm.side] = arm.pos - p
        return both(make)

    def carry(path, duration, label, hold_s=0.0):
        """Both arms carry the filter along ``path(s)`` -> pose, s 0..1; each corrects its command by its measured
        error (the arms sag under even a light load), and with ``hold_s`` holds the end until it is reached."""
        phase.update(left=label, right=label)
        n = max(1, round(duration / DT))
        for i in range(1, n + round(hold_s / DT) + 1):
            g = grips(path(smooth(min(i, n) / n)))
            for s in sides:
                err = g[s][0] - measured_tcp(s)[0]
                lead[s] = np.clip(lead[s] + CARRY_GAIN * err, -CARRY_LIMIT, CARRY_LIMIT)
                arms[s].pos = g[s][0] + lead[s]
                arms[s].rot = g[s][1]
            if i > n and all(np.linalg.norm(g[s][0] - measured_tcp(s)[0]) < 1.0 * MM for s in sides):
                break
            yield

    def let_go(label, away):
        def make(arm, y):
            yield from move(arm, arm.pos, 0.6, grip=OPEN, label=label)
            yield from move(arm, arm.pos + np.asarray(away, dtype=float), 1.0, label=label)
        return both(make)

    def cover_point(q, local):
        """A point of the cover (``local``: (x, z) on the shut cover, from the hinge), the cover at ``q``: (x, z)."""
        c, sn = math.cos(q), math.sin(q)
        x, z = local
        return hinge + np.array([x * c + z * sn, -x * sn + z * c])

    def turn(q, v):
        """A direction (x, z) on the shut cover, the cover at ``q``."""
        c, sn = math.cos(q), math.sin(q)
        return np.array([v[0] * c + v[1] * sn, -v[0] * sn + v[1] * c])

    def at(xz):
        return np.array([xz[0], COVER_Y, xz[1]])

    def follow(q0, q1, tcp, rot, duration, label):
        """The left arm's TCP along ``tcp(q)`` (and its rotation ``rot(q)``) as the cover goes from q0 to q1."""
        arm = arms["left"]
        phase["left"] = label
        n = round(duration / DT)
        for i in range(1, n + 1):
            q = q0 + (q1 - q0) * smooth(i / n)
            target = tcp(q)
            lead["cover"] = np.clip(lead["cover"] + CARRY_GAIN * (target - measured_tcp("left")[0]), -CARRY_LIMIT,
                                    CARRY_LIMIT)
            arm.pos = target + lead["cover"]
            arm.rot = rot(q)
            yield

    def approach(pos, rot, grip, back, label):
        """The left arm: come to ``pos`` from ``back`` away, line up, close in, line up."""
        arm = arms["left"]
        near = pos + np.asarray(back, dtype=float)
        yield from move(arm, near, 2.0, grip=grip, rot=rot, label=f"{label}: line up")
        yield from settle_until(arm, near, f"{label}: line up")
        yield from move(arm, pos + (arm.pos - near), 1.0, label=label)
        yield from settle_until(arm, pos, f"{label}: line up", tol=1.0 * MM, max_s=1.5)
        lead["cover"] = arm.pos - pos

    def script():
        if variant == "hold":
            phase.update(left="idle", right="idle")
            yield from idle()
        # 1. open the cover: pinch its finger pull and swing it out and up along its arc, let go, then push its
        # bottom edge up from below to the stop (its friction hinge keeps it where it is left)
        phase["right"] = "wait"
        arm = arms["left"]
        pull_local = (-ap.PANEL[2] - 1 * MM - PAD_TIP - ap.HINGE[0], ap.PANEL[0] + ap.PULL[1] / 2 - ap.HINGE[1])
        yield from approach(at(cover_point(0.0, pull_local)), fwd["left"], OPEN, (-0.05, 0.0, 0.0),
                            "open the cover: pinch its pull")
        yield from move(arm, arm.pos, 0.6, grip=0.0, label="open the cover: pinch its pull")
        q1 = math.radians(OPEN_FIRST_DEG)
        yield from follow(0.0, q1, lambda q: at(cover_point(q, pull_local)), lambda q: rot_y(q) @ fwd["left"], 2.0,
                          "open the cover: swing it out")
        yield from move(arm, arm.pos, 0.4, grip=OPEN, label="open the cover: let go")
        yield from move(arm, arm.pos - arm.rot[:, 2] * 0.03, 0.6, label="open the cover: let go")
        edge = (-ap.HINGE[0], ap.PANEL[0] - ap.HINGE[1])  # the cover's bottom edge, inner corner
        under = lambda q: at(cover_point(q, edge) + np.array([-8 * MM, -(FINGER_SIDE - PUSH_UP_PRESS)]))  # noqa: E731
        yield from approach(under(q1), fwd["left"], 0.0, (0.0, 0.0, -0.04), "open the cover: under its edge")
        yield from follow(q1, math.radians(ap.OPEN_DEG + 2.0), under, lambda q: fwd["left"], 2.5,
                          "open the cover: push it up")
        yield from move(arm, arm.pos + np.array([-0.03, 0.0, -0.06]), 1.0, label="away from the cover")
        report["cover_open_step"] = world["step"]
        # 2. take the dusty filter out: pinch its tabs, lift it off its lugs, swing its bottom out (its top still
        # behind the lip), lower its top clear of the lip, bring it out and lay it down face up near the robot
        yield from to_grasp(seated, "dusty filter", (-0.05, 0.0, 0.0))
        o0 = seated[0]
        top0 = o0 + p_top
        yield from carry(lambda s: (o0 + np.array([LIFT_BACK * s, 0.0, LIFT * s]), np.eye(3)), 0.8,
                         "lift it off the lugs", hold_s=1.5)
        top1 = top0 + np.array([LIFT_BACK, 0.0, LIFT])
        tilt = math.radians(TILT_DEG)
        yield from carry(lambda s: filt(tilt * s, p_top, top1), 1.2, "swing its bottom out")
        top2 = top1 + np.array([-0.012, 0.0, -DROP])
        yield from carry(lambda s: filt(tilt, p_top, top1 + s * (top2 - top1)), 1.2, "lower its top past the lip")
        tab2 = filt(tilt, p_top, top2)
        tab2 = tab2[0] + tab2[1] @ p_tab
        tab3 = tab2 + np.array([-0.10, 0.0, 0.0])
        yield from carry(lambda s: filt(tilt, p_tab, tab2 + s * (tab3 - tab2)), 1.2, "bring it out")
        tab_end = lying_dusty[0] + lying_dusty[1] @ p_tab
        yield from carry(lambda s: filt(tilt + (lie - tilt) * s, p_tab, tab3 + s * (tab_end + np.array([0, 0, 0.004]) - tab3)),
                         4.0, "lay it down")
        yield from let_go("let go", (0.0, 0.0, 0.05))
        report["dusty_out_step"] = world["step"]
        if variant == "outonly":
            phase.update(left="done", right="done")
            report["script_done_step"] = world["step"]
            yield from idle()
        # 3. pick up the clean filter and turn it up, its top toward the unit
        yield from to_grasp(lying_clean, "clean filter", (0.0, 0.0, 0.05))
        tabc = lying_clean[0] + lying_clean[1] @ p_tab
        # the fitting: its top just under the lip, behind it, tilted bottom out; then up behind the lip, swung
        # upright (its bottom over the lugs), and down onto the ledge
        a1 = math.radians(FIT_TILT_DEG)
        top_a = np.array([F + FIT_TOP_X, 0.0, T + ap.LIP_Z - 4 * MM])
        pose_a = filt(a1, p_top, top_a)
        tab_a = pose_a[0] + pose_a[1] @ p_tab
        tab_m = np.array([tab_a[0] - 0.10, 0.0, tab_a[2] - 0.02])
        yield from carry(lambda s: filt(lie + (a1 - lie) * s, p_tab, tabc + s * (tab_m - tabc) + np.array([0, 0, 0.08 * math.sin(math.pi * s)])),
                         4.0, "turn it up")
        yield from carry(lambda s: filt(a1, p_tab, tab_m + s * (tab_a - tab_m)), 1.5, "bring it to the unit")
        top_b = top_a + np.array([0.0, 0.0, FIT_RAISE])
        yield from carry(lambda s: filt(a1, p_top, top_a + s * (top_b - top_a)), 1.0, "its top up behind the lip",
                         hold_s=1.0)
        if variant == "notfit":  # left there, its top behind the lip, still tilted
            yield from let_go("let go", (-0.06, 0.0, 0.0))
            phase.update(left="done", right="done")
            report["script_done_step"] = world["step"]
            yield from idle()
        up = seated[0] + np.array([FIT_DEEPER, 0.0, FIT_ABOVE])
        top_c = up + p_top
        yield from carry(lambda s: filt(a1 * (1 - s), p_top, top_b + s * (top_c - top_b)), 1.5, "swing it upright",
                         hold_s=1.0)
        down = seated[0] + np.array([FIT_DEEPER, 0.0, 0.0])
        yield from carry(lambda s: (up + s * (down - up), np.eye(3)), 1.0, "down onto the ledge")
        yield from let_go("let go", (-0.06, 0.0, -0.01))
        report["clean_in_step"] = world["step"]
        if variant == "open":
            phase.update(left="done", right="done")
            report["script_done_step"] = world["step"]
            yield from idle()
        # 4. close the cover: going round its bottom edge, press its face down from above, then from in front push
        # it shut with the fingertips
        arm = arms["left"]
        clear_x = F + ap.HINGE[0] - (ap.HINGE[1] - ap.PANEL[0]) - 0.04  # in front of its bottom edge, whatever its angle
        face_a = (-ap.PANEL[2] - ap.HINGE[0], -CLOSE_AT)  # a point on its face (from the hinge)
        out_n = (-1.0, 0.0)  # its face's outward normal (shut)
        above = lambda q: at(cover_point(q, face_a) + turn(q, out_n) * (FINGER_SIDE - PRESS))  # noqa: E731
        q_open, q_mid = math.radians(ap.OPEN_DEG), math.radians(CLOSE_MID_DEG)
        top = above(q_open)
        yield from move(arm, np.array([clear_x, COVER_Y, arm.pos[2]]), 1.2, grip=0.0, rot=fwd["left"],
                        label="close the cover: in front of it")
        yield from move(arm, np.array([clear_x, COVER_Y, top[2] + 0.02]), 1.0, label="close the cover: up")
        yield from approach(top, fwd["left"], 0.0, (0.0, 0.0, 0.015), "close the cover: on its face")
        yield from follow(q_open, q_mid, above, lambda q: fwd["left"], 2.0, "close the cover: press it down")
        yield from move(arm, arm.pos + np.array([0.0, 0.0, 0.02]), 0.5, label="close the cover: off it")
        face_b = (-ap.PANEL[2] - ap.HINGE[0], -CLOSE_TIPS_AT)
        tips = lambda q: at(cover_point(q, face_b) + np.array([-(PAD_TIP - PRESS), 0.0]))  # noqa: E731
        start = tips(q_mid)
        yield from move(arm, np.array([clear_x, COVER_Y, arm.pos[2]]), 1.0, label="close the cover: in front of it")
        yield from move(arm, np.array([clear_x, COVER_Y, start[2]]), 1.0, label="close the cover: down")
        yield from approach(start, fwd["left"], 0.0, (-0.03, 0.0, 0.0), "close the cover: tips on its face")
        yield from follow(q_mid, math.radians(-3.0), tips, lambda q: fwd["left"], 2.0, "close the cover: push it shut")
        arm = arms["left"]
        yield from move(arm, arm.pos + np.array([-0.06, 0.0, 0.03]), 1.0, label="away")
        phase.update(left="done", right="done")
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
        a = pack(arms["left"].action(), arms["right"].action())
        res = emb.step(Action(data=a))
        world["obs"] = res.observation
        world["states"] = object_states(emb._env, emb.task.names)
        st = world["states"]
        o_d, r_d = ap.frame(st["filter_dusty"])
        o_c, r_c = ap.frame(st["filter_clean"])
        row = {"step": step, "left": phase["left"], "right": phase["right"], "score": res.info.get("score"),
               "success": res.info.get("success"), "cover_deg": round(st["unit"].cover_deg, 2),
               "dusty_in": task.in_unit(st["filter_dusty"]), "clean_fitted": task.fitted(st["filter_clean"]),
               "dusty": [round(float(v), 4) for v in o_d], "clean": [round(float(v), 4) for v in o_c],
               "dusty_deg": round(float(math.degrees(math.atan2(r_d[0, 2], r_d[2, 2]))), 2),
               "clean_deg": round(float(math.degrees(math.atan2(r_c[0, 2], r_c[2, 2]))), 2)}
        jl, jr = split(np.asarray(world["obs"].state["joint_pos"]))
        row["grip"] = [round(float(j[6]), 3) for j in (jl, jr)]  # each gripper: 0 shut, 1 open
        for side in ("left", "right"):
            row[f"{side[0]}_cmd"] = [round(float(v), 4) for v in arms[side].pos]
            row[f"{side[0]}_meas"] = [round(float(v), 4) for v in measured_tcp(side)[0]]
            row[f"{side[0]}_ik"] = [round(float(v), 4) for v in getattr(arms[side], "ik_err", (0.0, 0.0))]
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
        if res.terminated or done_script:
            result = res
            break
        result = res
    success = bool(result.terminated and result.termination_reason == "success")
    st = world["states"]
    summary = {
        "variant": variant,
        "task_score": 1.0 if success else float(result.info.get("score", 0.0)),
        "terminated": bool(result.terminated),
        "termination_reason": result.termination_reason,
        "steps": len(log),
        "run_limit_steps": MAX_STEPS,
        "cover_deg": round(st["unit"].cover_deg, 2),
        "dusty_in": task.in_unit(st["filter_dusty"]),
        "clean_fitted": task.fitted(st["filter_clean"]),
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
                    help="physics device; 'cpu' steps this one-environment scene ~3x faster (CPU PhysX), but its "
                         "results differ in detail from the GPU solver trials run on: confirm on cuda:0")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override a constant for an experiment (a Python literal): --set OPEN=0.4, or a part's with --set tp.PAD_SPRING=4000.0")
    args = ap.parse_args(argv)
    for item in args.set:
        name, value = item.split("=", 1)
        if name.startswith("tp."):  # a part constant: set on tb_parts (which task.py imports) before any USD is written
            sys.path.insert(0, str(HERE))
            import tb_parts

            target = vars(tb_parts)
        else:
            target = globals()
        name = name.removeprefix("tp.")
        if name not in target:
            ap.error(f"--set: no constant {name!r}")
        target[name] = ast.literal_eval(value)
    video = not args.no_video
    emb = make_embodiment(video, args.device)
    for variant in VARIANTS if args.variant == "all" else (args.variant,):
        out = args.out / variant if args.out and args.variant == "all" else args.out or HERE / "runs" / variant
        summary = run(variant, out, video, args.max_steps, emb)
        print("RESULT " + json.dumps({k: summary[k] for k in ("variant", "task_score", "steps", "termination_reason",
                                                               "cover_deg", "dusty_in", "clean_fitted", "wall_s")}))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
