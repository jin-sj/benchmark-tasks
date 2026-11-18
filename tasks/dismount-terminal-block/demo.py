"""Scripted two-arm demo for the dismount task, closed loop on the simulator's object poses.

    python demo.py                    # left arm holds the rail; right arm takes the blue block off, into the tray: 1.0
    python demo.py --variant held     # the same, but the block is lowered onto the tray floor and never let go: 0.75
    python demo.py --variant lifted   # the same, but the block is held up above the rail after lifting it off: 0.5
    python demo.py --variant lever_only  # only the screwdriver: the foot levered out and held there, the block left
                                         # on the rail: 0.25
    python demo.py --variant no_lever    # the same arm motions as the full demo without the screwdriver: 0
    python demo.py --device cpu       # CPU physics, ~3x faster, for quick checks

It drives the ``yam_arena`` embodiment directly (20 Hz control, object_jitter 0, as a trial runs) and reads the
task's own scoring from each step's ``info``:

- left arm: fingers closed, pressed into the rail's channel 17 cm left of the rail's centre (between the left beige
  block and the left end bracket), for the whole run;
- right arm: grips the blue block across its broad faces; pushes it toward the arms (the command leading) until it
  has moved 2.4 mm, the fixed hook giving, so the sprung foot comes out from under its lip; tips the near end up a few
  degrees along an arc about the hook; slides it toward the far side until the hook is out; lifts it clear; carries
  it to the tray and lets go just above the floor.

Writes ``result.json``, ``steps.jsonl`` and (unless ``--no-video``) one video per camera to ``--out``.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TASK = HERE / "task.py"
INSTRUCTION = (
    "Take the blue terminal block off the DIN rail with the screwdriver and put it in the tray, and leave the two beige "
    "blocks where they are."
)
VARIANTS = ("full", "held", "lifted", "no_lever", "lever_only")
CAMERAS = ("top_cam", "left_cam", "right_cam")
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ
MAX_STEPS = 2400  # the task's run limit (the screwdriver release takes longer than the old push)
MM = 1e-3

PAD_TIP = 0.013  # pad tips beyond the TCP
OPEN = 0.35  # gripper opening (0..1 of 94 mm): ~33 mm between the pads
GRASP_DEPTH = 18 * MM  # pad tips this far below the block's top
GRIP_BACK = 0 * MM  # (unused)
GRIP_FAR = 30 * MM  # grip this far from the block's origin toward its far (hook) end: clear of the screwdriver at the near end
HOLD_AT = 170 * MM  # the left arm holds the rail here (rail frame y): 15 cm from the blue block, so the two grippers
# (each ~10 cm wide) stay apart. Pulling the block up tips the rail about its left end: the nearer the hold is to the
# block (the further from that end), the better it resists; 17 cm let the rail tip, 13 cm held it
HOLD_OPEN = 0.48  # ~45 mm between the pads, around the rail's 35 mm lips
HOLD_DEPTH = 4 * MM  # pad tips this far below the lip tops (the rail is 7.5 mm tall)
HOLD_LEAN = 0.0  # upright: a leaned press also pushes the rail along
HOLD_STYLE = "press"  # "block": press the block's far end down; "press": fingertips into the rail's channel; "grip"
PRESS_AT = 18 * MM  # "block": the left fingertips this far from the block's middle, on its far (hook) end
PRESS_LEAN = 30.0  # degrees: the left gripper leaned so its tips point back toward the arms and its wrist is beyond
BLOCK_PRESS = 6 * MM  # "block": the fingertips commanded this far into the block's top  # "grip": the rail's lip edges between the fingers; "press": fingertips pressed into its channel
HOLD_PRESS = 40 * MM  # "press": commanded this far into the rail (the arm is soft: this sets the press force)
HOLD_X = 8 * MM  # "press": this far from the centreline toward the arms, in the channel (inside the rail's footprint:
# pressing an overhanging lip tips the rail; off-centre toward the arms resists the roll a pull on the block causes)
PUSH_RATE, PUSH_MAX = 10 * MM, 40 * MM  # the push command leads by this much more per s, at most this far
PUSHED = 2.4 * MM  # pushed this far toward the arms: the sprung foot's tooth is out past its lip's edge
TIP_DEG = 6.0  # then tip the near end up this much about the hook (just enough for the sprung foot to clear its lip)
TIP_RATE_DEG = 6.0  # per second
UNHOOKED = 3.5 * MM  # then slide toward the far side until the block is this far beyond where it started (the
# hook's 2.5 mm tooth is out from under its lip)
SLIDE_RATE, SLIDE_MAX = 10 * MM, 40 * MM
TRAY_DROP = 15 * MM  # let go with the block's lowest point this far above the tray floor
# the screwdriver (left arm)
SD_OPEN = 0.35  # ~33 mm: open round the 20 mm handle
SD_GRIP_BELOW_TOP = 12 * MM  # pad tips this far below the handle's top
SD_LIFT = 0.07  # lifted this far out of the holder
SLOT_ABOVE = 12 * MM  # the blade's tip first this far above the slot's floor ...
SLOT_IN = 1.5 * MM  # ... then down to this
SD_CENTRE_TOL, SD_CENTRE_TRIES = 0.25 * MM, 6
LEVER_OPEN = 2.4 * MM  # lever the sprung foot out this far (its tooth clears the lip's edge at 2.0 mm)
LEVER_RATE, LEVER_MAX = 3 * MM, 30 * MM  # the blade's push toward the arms leads (the arm is soft; 12 mm opened
# the foot only 1 mm, run 6)
LEVER_S = 5.0  # the lever's push lasts at most this long
# lever_only: lever until the foot's tooth is this far out in the rail's frame, the push lasting at most this long
LEVER_ONLY_OPEN = float(os.environ.get("DTB_LEVER_ONLY_OPEN_MM", "2.4")) * MM
LEVER_ONLY_S = float(os.environ.get("DTB_LEVER_ONLY_S", "12.0"))
LEVER_ONLY_MAX = float(os.environ.get("DTB_LEVER_ONLY_MAX_MM", "40")) * MM
GRIP_LEAN = math.radians(20.0)  # the right hand leans so its body is beyond the block's far end (upright, it met the
# left hand's screwdriver over the block, runs 3-6)
SD_FOLLOW = 8 * MM  # while the right arm tips the block, the blade follows the near end up this far
SD_LEAN = math.radians(6.0)  # the screwdriver held leaning away from the block (its handle toward the arms): the two
# hands met over the block when it was upright, runs 3-5
AFTER_SUCCESS_S = 3.0


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
    obs = emb.reset(Scene(id=f"dismount-{variant}", instruction=INSTRUCTION), seed=0)
    set_rendering(emb, video)
    task = emb.task.module
    tp = task.tp
    from yam_arena.scene import object_states  # patched by the task: the jointed blocks

    left_q, right_q = split(np.asarray(obs.state["joint_pos"]))
    arms = {
        "left": Arm("left", station.BASES["left"], left_q[:6], ik, grip_rotation),
        "right": Arm("right", station.BASES["right"], right_q[:6], ik, grip_rotation),
    }
    world = {"states": object_states(emb._env, emb.task.names), "obs": obs, "step": 0}
    report = {}
    flags = set()

    def wait_for(arm, flag: str, label: str):
        phase[arm.side] = label
        while flag not in flags:
            yield

    def rail_point(local) -> np.ndarray:
        p, r = task.rail_pose(world["states"])
        return p + r @ np.asarray(local, dtype=float)

    def rail_yaw() -> float:
        _, r = task.rail_pose(world["states"])
        return math.atan2(r[1, 0], r[0, 0])

    def block(name=task.TARGET):
        return task._origin(world["states"][name], task.C_BLOCK)

    def on_rail(name=task.TARGET):
        return task.block_on_rail(world["states"], name)

    def measured_tcp(side: str):
        left, right = split(np.asarray(world["obs"].state["joint_pos"]))
        q = (left if side == "left" else right)[:6]
        return ik.tcp_pose(q, arms[side].base)

    phase = {"left": "", "right": ""}

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

    def settle(arm: Arm, target, duration: float, label: str, gain=0.5, limit=0.015):
        """Hold ``target``, correcting the command by the measured TCP error each step."""
        phase[arm.side] = label
        target = np.asarray(target, dtype=float)
        corr = np.zeros(3)
        for _ in range(max(1, round(duration / DT))):
            corr = np.clip(corr + gain * (target - measured_tcp(arm.side)[0]), -limit, limit)
            arm.pos = target + corr
            yield

    # -- left arm: the screwdriver ---------------------------------------------------------------------------
    def driver():
        """(blade tip, rotation) of the screwdriver (its origin is the blade's tip)."""
        return task._origin(world["states"]["screwdriver"], np.array(tp.local_center("screwdriver")))

    def slot_point(z: float) -> np.ndarray:
        """The middle of the blue block's release slot, z above the slot's floor (world)."""
        o, r = block()
        x = -(tp.HOOK_SLOT_CHEEK[0] + tp.SLOT_WIDTH / 2)
        return o + r @ np.array([x, 0.0, tp.SLOT_FLOOR + z])

    def tip_to(arm: Arm, target, tries: int, label: str, xy_only=False):
        """Bring the blade's tip to ``target`` by the measured tip (correct, settle, measure)."""
        err = np.zeros(3)
        for k in range(tries):
            tip, _ = driver()
            err = np.asarray(target) - tip
            if xy_only:
                err[2] = 0.0
            if np.linalg.norm(err[:2]) < SD_CENTRE_TOL and (xy_only or abs(err[2]) < SD_CENTRE_TOL):
                break
            yield from move(arm, arm.pos + err, 0.5, label=label)
            yield from move(arm, arm.pos, 0.4, label=label + ": settle")
        return float(np.linalg.norm(err[:2]))

    def left_script():
        arm = arms["left"]
        rep = report.setdefault("screwdriver", {})
        if variant == "no_lever":  # the screwdriver stays in its holder
            phase["left"] = "idle"
            flags.add("foot_open")
            flags.add("left_clear")
            while True:
                yield
        tip, r = driver()
        top = tip + r @ np.array([0.0, 0.0, tp.SHAFT[1] + tp.HANDLE[1]])
        g = top - np.array([0.0, 0.0, SD_GRIP_BELOW_TOP]) + np.array([0.0, 0.0, PAD_TIP])
        hand = rot_z(math.pi / 2) @ arm.down  # fingers closing across the rail: the hand narrow toward the block
        yield from move(arm, g + (0, 0, 0.05), 2.5, grip=SD_OPEN, rot=hand, label="over the screwdriver")
        yield from move(arm, g, 1.2, label="down round its handle")
        yield from settle(arm, g, 0.6, "line up on the handle")
        yield from move(arm, arm.pos, 0.6, grip=0.0, label="grip the handle")
        yield from move(arm, arm.pos + (0, 0, SD_LIFT), 1.2, label="lift it out of the holder")
        yield from move(arm, arm.pos, 0.8, rot=rot_y(-SD_LEAN) @ arm.rot, label="lean it away from the block")
        rep["picked"] = bool(driver()[0][2] > tip[2] + 0.04)
        # (first, before the right hand comes: coming down beside the right hand's grip, the two hands met and the
        # screwdriver was knocked over in the grip, run 2)
        # over the release slot, the blade's tip lined up on the slot's middle by its measured position
        tip, _ = driver()
        above = slot_point(SLOT_ABOVE)
        yield from move(arm, arm.pos + (above - tip) + (0, 0, 0.03), 2.0, label="over the release slot")
        yield from move(arm, arm.pos - (0, 0, 0.03), 1.0, label="down over the slot")
        rep["centre_err_mm"] = round((yield from tip_to(arm, slot_point(SLOT_ABOVE), SD_CENTRE_TRIES,
                                                        "line the blade up over the slot")) / MM, 3)
        # into the slot, keeping it lined up
        tip, _ = driver()
        yield from move(arm, arm.pos + (slot_point(SLOT_IN) - tip), 1.2, label="blade into the slot")
        yield from tip_to(arm, slot_point(SLOT_IN), 2, "line up in the slot", xy_only=True)
        yield from move(arm, arm.pos, 0.4, label="settle in the slot")
        tip, _ = driver()
        rep["in_slot_mm"] = [round(float(v) / MM, 2) for v in (tip - slot_point(0.0))]
        # lever: the blade pushes the foot's tab toward the arms (the command leads) until the foot is out
        base_pos = arm.pos.copy()
        lead = 0.0
        phase["left"] = "lever the foot out"
        only = variant == "lever_only"
        for _ in range(round((LEVER_ONLY_S if only else LEVER_S) / DT)):
            if (task.foot_out(on_rail()) >= LEVER_ONLY_OPEN) if only else (on_rail()["feet_spread"][1] >= LEVER_OPEN):
                break
            lead = min(lead + LEVER_RATE * DT, LEVER_ONLY_MAX if only else LEVER_MAX)
            arm.pos = base_pos - np.array([lead, 0.0, 0.0])
            yield
        rep["lever_lead_mm"], rep["foot_spread_mm"] = round(lead / MM, 1), round(on_rail()["feet_spread"][1] / MM, 2)
        rep["foot_out_mm"] = round(task.foot_out(on_rail()) / MM, 2)
        rep["open_step"] = world["step"]
        if only:  # hold the foot out with the blade; the right arm never comes: the block stays on the rail
            phase["left"] = "hold the foot out"
            report["script_done_step"] = world["step"] + round(2.0 / DT)
            while True:
                yield
        flags.add("foot_open")
        yield from wait_for(arm, "gripped", "hold the foot out")
        # follow the near end up while the right arm tips the block (the foot stays out), then pull the blade out
        yield from wait_for(arm, "tipping", "hold the foot out")
        yield from move(arm, arm.pos + (0, 0, SD_FOLLOW), 1.3, label="follow the near end up")
        yield from wait_for(arm, "tipped", "hold")
        yield from move(arm, arm.pos + (0.01, 0, 0.04), 1.0, label="blade out of the slot")
        yield from move(arm, arm.pos + (-0.04, 0.10, 0.02), 1.5, label="out of the way")
        flags.add("left_clear")
        phase["left"] = "done"
        while True:
            yield

    # -- right arm: take the blue block off, into the tray -------------------------------------------------
    def right_script():
        arm = arms["right"]
        o, r = block()
        g = o + r @ np.array([GRIP_FAR, 0.0, tp.BLOCK_HEIGHT - GRASP_DEPTH]) + np.array([0, 0, PAD_TIP])
        # 1. the left arm levers the sprung foot out with the screwdriver first (the right arm waits where it starts:
        # waiting over the block, the two hands met, run 3)
        yield from wait_for(arm, "foot_open", "wait for the foot to be levered out")
        if variant == "lever_only":
            phase["right"] = "idle"
            while True:
                yield
        o, r = block()
        lean = rot_y(GRIP_LEAN) @ arm.down
        approach = lean[:, 2]
        g = o + r @ np.array([GRIP_FAR, 0.0, tp.BLOCK_HEIGHT - GRASP_DEPTH]) - PAD_TIP * approach
        yield from move(arm, g + (0, 0, 0.06), 2.5, grip=OPEN, rot=lean, label="over the blue block")
        yield from move(arm, g + (0, 0, 0.012), 1.2, label="down near it")
        yield from settle(arm, g + (0, 0, 0.012), 0.8, "line up over it")
        corr = arm.pos - (g + (0, 0, 0.012))
        yield from move(arm, g + corr, 0.6, label="down to it")
        yield from settle(arm, g, 0.5, "line up over it")
        yield from move(arm, arm.pos, 0.8, grip=0.0, label="grip its broad faces")
        flags.add("gripped")
        flags.add("tipping")
        lead = 0.0
        # 2. tip the near end up a little, along an arc about the hook (still pushing: the lead stays)
        phase["right"] = "tip the near end up"
        for _ in range(round(4.0 / DT)):
            if on_rail()["tilt_deg"] >= TIP_DEG:
                break
            o, r = block()
            pivot = o + r @ np.array([tp.LIP_EDGE - 2.5 * MM, 0.0, -tp.LIP_THICK])
            step_a = math.radians(TIP_RATE_DEG) * DT
            c, sn = math.cos(step_a), math.sin(step_a)
            rel = arm.pos - pivot
            arm.pos = pivot + np.array([c * rel[0] + sn * rel[2], rel[1], -sn * rel[0] + c * rel[2]])
            yield
        report["tipped_step"], report["tip_deg"] = world["step"], round(on_rail()["tilt_deg"], 1)
        flags.add("tipped")
        yield from wait_for(arm, "left_clear", "wait for the screwdriver to be clear")
        # 3. slide it toward the far side until the hook is out from under its lip, the command leading
        phase["right"] = "slide toward the far side"
        start = arm.pos.copy()
        lead = 0.0
        for _ in range(round(6.0 / DT)):
            if on_rail()["across"] >= UNHOOKED:
                break
            lead = min(lead + SLIDE_RATE * DT, SLIDE_MAX)
            arm.pos = start + np.array([lead, 0.0, 0.0])
            yield
        report["unhooked_step"], report["slide_lead_mm"] = world["step"], round(lead / MM, 1)
        report["unhooked_mm"] = round(on_rail()["across"] / MM, 2)
        # 4. lift it clear (take the lead back out first, so it comes straight up)
        yield from move(arm, arm.pos + (-lead + 2 * MM, 0, 0.06), 1.5, label="lift it clear")
        report["off_step"] = world["step"]
        if variant == "lifted":  # hold it up above the rail, off it but never carried to the tray
            phase["right"] = "hold it above the rail"
            report["script_done_step"] = world["step"] + round(3.0 / DT)
            while True:
                yield
        # to the tray
        spot = np.array([task.TRAY_ORIGIN[0], task.TRAY_ORIGIN[1], task.TOP + tp.TRAY_FLOOR])
        yield from move(arm, (spot[0], spot[1], arm.pos[2] + 0.02), 2.5, rot=arm.down, label="carry it")
        low = float(world["states"][task.TARGET].low[2])
        drop = arm.pos[2] - (low - (spot[2] + TRAY_DROP))
        yield from move(arm, (arm.pos[0], arm.pos[1], drop), 1.5, label="lower it")
        if variant == "held":  # lower it onto the tray floor and never let go: in the tray, but not released
            yield from move(arm, arm.pos - np.array([0.0, 0.0, TRAY_DROP - 1 * MM]), 1.0, label="down onto the floor")
            phase["right"] = "hold it in the tray"
            report["script_done_step"] = world["step"] + round(3.0 / DT)
            while True:
                yield
        yield from move(arm, arm.pos, 0.6, grip=OPEN, label="let go")
        yield from move(arm, arm.pos + (0, 0, 0.06), 1.0, label="up")
        yield from move(arm, arm.pos + (-0.05, 0.05, 0.0), 1.2, label="away")
        phase["right"] = "done"
        report["script_done_step"] = world["step"]
        while True:
            yield

    scripts = {"left": left_script(), "right": right_script()}
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
        for side in ("left", "right"):
            next(scripts[side])
        a = pack(arms["left"].action(), arms["right"].action())
        res = emb.step(Action(data=a))
        world["obs"] = res.observation
        world["states"] = object_states(emb._env, emb.task.names)
        st = world["states"]
        shift, turn = task.rail_motion(st)
        _, rr = task.rail_pose(st)
        rpy = [round(math.degrees(v), 2) for v in (math.atan2(rr[2, 1], rr[2, 2]), -math.asin(max(-1.0, min(1.0, rr[2, 0]))),
                                                   math.atan2(rr[1, 0], rr[0, 0]))]
        sd_tip, sd_rot = driver()
        row = {"sd_tip": [round(float(v), 4) for v in sd_tip], "sd_tilt_deg": round(math.degrees(math.acos(min(1.0, abs(sd_rot[2, 2])))), 2),
               "slot": [round(float(v), 4) for v in slot_point(0.0)],
               "l_cmd": [round(float(v), 4) for v in arms["left"].pos], "l_meas": [round(float(v), 4) for v in measured_tcp("left")[0]],
               "l_ik_mm": round(float(arms["left"].ik_err[0]) * 1000, 2) if hasattr(arms["left"], "ik_err") else None}
        row.update({"step": step, "rail_rpy_deg": rpy, "left": phase["left"], "right": phase["right"], "score": res.info.get("score"),
               "success": res.info.get("success"), "stage": task.stage(st), "undisturbed": task.undisturbed(st),
               "rail_shift_mm": round(shift / MM, 3), "rail_turn_deg": round(turn, 3),
               "blue_low_z": round(float(st[task.TARGET].low[2]), 4),
               "blue_contact_n": round(float(st[task.TARGET].robot_contact_n), 3),
               "blue_on_floor": task.on_tray_floor(st[task.TARGET]), "blue_released": task.released(st[task.TARGET])})
        for n in task.BLOCKS:
            b = on_rail(n)
            row[n] = {"along_mm": round(b["along"] / MM, 2), "across_mm": round(b["across"] / MM, 2),
                      "h_mm": round(b["height"] / MM, 2), "feet_mm": [round(f / MM, 2) for f in b["feet"]],
                      "spread_mm": [round(t / MM, 2) for t in b["feet_spread"]], "tilt": round(b["tilt_deg"], 2),
                      "latched": task.latched(b), "foot_out_mm": round(task.foot_out(b) / MM, 2)}
        for side in ("left", "right"):
            row[f"{side[0]}_cmd"] = [round(float(v), 4) for v in arms[side].pos]
            row[f"{side[0]}_meas"] = [round(float(v), 4) for v in measured_tcp(side)[0]]
        log.append(row)
        if video:
            for c in CAMERAS:
                frames[c].append(res.observation.images[c])
        done_script = "script_done_step" in report and step - report["script_done_step"] >= round(3.0 / DT)
        if "rail_moved_step" not in report and not task.rail_ok(st):
            report["rail_moved_step"] = step
        done_script = done_script or ("rail_moved_step" in report and step - report["rail_moved_step"] >= round(1.0 / DT))
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
    st = world["states"]
    shift, turn = task.rail_motion(st)
    success = bool(result.terminated and result.termination_reason == "success")
    summary = {
        "variant": variant,
        "task_score": 1.0 if success else float(result.info.get("score", 0.0)),
        "terminated": bool(result.terminated),
        "termination_reason": result.termination_reason,
        "steps": len(log),
        "run_limit_steps": MAX_STEPS,
        "final_stage": task.stage(st),
        "undisturbed": task.undisturbed(st),
        "rail_shift_mm": shift / MM,
        "rail_turn_deg": turn,
        "max_rail_shift_mm": max(r["rail_shift_mm"] for r in log),
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
                                                               "final_stage", "undisturbed", "rail_shift_mm",
                                                               "wall_s")}))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
