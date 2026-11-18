"""Scripted two-arm demo for the dismount task: the screwdriver release, then the block taken off by hand.

    python demo.py                        # full: the block released into the tray and let go: 1.0
    python demo.py --variant held         # the same, but lowered onto the tray floor and never let go: 0.75
    python demo.py --variant lifted       # the same until it is off its hook, then held 60 mm above the rail: 0.5
    python demo.py --variant lever_only   # the screwdriver levers the foot out and holds it there: 0.25
    python demo.py --variant no_lever     # the right arm's motions without the screwdriver: 0 (the block stays latched)

It drives the ``yam_arena`` embodiment directly (20 Hz control, object_jitter 0, as a trial runs) and reads the
task's own score from each step's ``info``. The script plans from the known starting layout (task.py) and the
arms' joint readings only: the screwdriver's blade tip is where the layout puts it, carried by the left hand as
gripped; the block is where the layout puts it. Object poses from the simulation are only written to the log.

- left arm: takes the screwdriver from its holder; brings the blade over the blue block's release slot and lowers it
  in (if it lands on the slot's edge, the hand stops high: it lifts and tries 0.4 mm to either side); levers: tips
  the handle toward the block about the slot's top edge and pushes it toward the arms, the command leading, so the
  blade pushes the foot's tab and the foot out from under the near lip; draws the screwdriver up a few millimetres,
  so the near end comes up with it (the foot's tooth then rests on the lip's edge); takes the blade out, moves away;
- right arm, then: grips the blue block across its broad faces (the fingertips in the 10 mm gaps beside it); tips the
  near end up along an arc about the hook; slides it toward the far side until the hook is out from under its lip;
  lifts it clear; carries it to the tray and lets go just above the floor.

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
INSTRUCTION = (
    "Take the blue terminal block off the DIN rail with the screwdriver and put it in the tray, and leave the two beige "
    "blocks where they are."
)
VARIANTS = ("full", "held", "lifted", "lever_only", "no_lever", "probe", "endgrip")
CAMERAS = ("top_cam", "left_cam", "right_cam")
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ
MAX_STEPS = 2400  # the task's run limit (the screwdriver release takes longer than the old push)
MM = 1e-3

PAD_TIP = 0.013  # pad tips beyond the TCP
OPEN = 0.10  # gripper opening (0..1 of 94 mm): 9.4 mm between the pads, 1.6 mm clear of each broad face; the fingertips
# (6.3 mm thick) then stand 2 mm clear of the beige blocks 10 mm away
GRASP_DEPTH = 11 * MM  # pad tips this far below the block's top (deeper, the finger thickens to 10 mm: it would meet the beige blocks)
GRIP_FAR = 5 * MM  # grip this far from the block's origin toward its far (hook) end (30 mm: the block turned in the grip, run 29)
TIP_RATE_DEG = 6.0  # per second
SLIDE_RATE, SLIDE_LEAD = 10 * MM, 15 * MM  # the slide toward the far side: per second, in all (command, leading)
TRAY_DROP = 15 * MM  # let go with the block's lowest point this far above the tray floor
# the screwdriver (left arm)
SD_OPEN = 0.35  # ~33 mm: open round the 20 mm handle
SD_GRIP_BELOW_TOP = 12 * MM  # pad tips this far below the handle's top
SD_LIFT = 0.07  # lifted this far out of the holder
SLOT_ABOVE = 12 * MM  # the blade's tip first this far above the slot's floor ...
SLOT_IN = 1.5 * MM  # ... then down to this
SLOT_TRIES = (0.0, -0.4 * MM, 0.4 * MM, -0.8 * MM)  # the blade tried at the slot's middle, then this far toward the
# cheek (negative: toward the tab), until it goes in
SLOT_IN_OK = 4 * MM  # in: its tip less than this above the slot's floor (on the slot's edge it stays at 9 mm)
SD_CENTRE_TOL, SD_CENTRE_TRIES = 0.25 * MM, 6
GRIP_LEAN = math.radians(20.0)  # the right hand leans so its body is beyond the block's far end (upright, it met the
# left hand's screwdriver over the block, runs 3-6)
SD_LEAN = math.radians(6.0)  # the screwdriver held leaning away from the block (its handle toward the arms): the two
# hands met over the block when it was upright, runs 3-5
AFTER_SUCCESS_S = 3.0
POP_UP, POP_RATE = 8 * MM, 5 * MM  # with the foot levered out, the screwdriver is drawn up this far (command), per second
SD_FRONT = (0.0, 0.0, 0.03)  # the screwdriver comes over the slot from this much above it
HOVER = (0.03, 0.0, 0.15)  # the right hand comes in this far beyond and above its grip (at least at its start height)
LEVER_RATE_DEG, LEVER_MAX_DEG = 5.0, 12.0  # the lever's tilt toward the block: per second, at most (the arm is soft;
# beyond ~14 degrees the left wrist's pitch reaches its limit)
LEVER_PUSH_RATE, LEVER_PUSH_MAX = 4.0 * MM, 26 * MM  # ... and the hand pushed toward the arms: per second, at most
LEVER_S = 7.5  # the lever's ramp lasts at most this long
TIP_CMD_DEG = 15.0  # the right hand's arc about the hook (command; the near end rises ~7 mm, the foot clear of its lip)
SLOT_Y_GAIN = 0.01  # per step: the blade's tip kept on the slot's middle along the rail while levering
BLADE_OUT = (-0.02, 0.0, 0.03)  # the blade drawn out of the slot by this much (m)
ENDGRIP_X, ENDGRIP_Z, ENDGRIP_OPEN = 0.8 * MM, 3.0 * MM, 0.92  # (experiment 'endgrip')


def smooth(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


class Arm:
    """One arm's commanded TCP pose and gripper, turned into joint targets by IK (warm-started)."""

    def __init__(self, side: str, base, q0, ik, grip_rotation):
        self.side, self.base, self.ik = side, np.asarray(base, dtype=float), ik
        self.q = np.array(q0, dtype=float)
        self.q0 = self.q.copy()
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.grip = 1.0
        self.down = grip_rotation(0.0)

    def action(self) -> np.ndarray:
        quat = self.ik.matrix_to_quat_xyzw(self.rot)
        q, pe, re = self.ik.solve(self.pos, quat, self.q, base_pos=self.base, iters=60)
        if pe > 1e-3 or re > 0.02:  # the warm start got stuck (a joint at its limit): try other seeds, keep the best
            for seed in (self.q0, np.array([0.0, 1.0, 1.0, -0.6, 0.0, 0.0])):
                q2, pe2, re2 = self.ik.solve(self.pos, quat, seed, base_pos=self.base, iters=200)
                if pe2 + 0.05 * re2 < pe + 0.05 * re and np.max(np.abs(q2 - self.q)) < 0.3:
                    q, pe, re = q2, pe2, re2
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

    def true_block(name=task.TARGET):  # (for the log only)
        return task._origin(world["states"][name], task.C_BLOCK)

    def block(name=task.TARGET):
        """The block's pose from the known starting layout (latched on the rail), as the script plans with it."""
        return np.array([task.RAIL_ORIGIN[0], task.RAIL_ORIGIN[1] + task.START[name], task.SEATED_Z]), np.eye(3)

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
    def true_driver():  # (for the log only)
        return task._origin(world["states"]["screwdriver"], np.array(tp.local_center("screwdriver")))

    grasp = {"off": None, "rot": np.eye(3)}

    def driver():
        """(blade tip, rotation) of the screwdriver, from the left arm's joint readings: before it is gripped, where
        the layout puts it (upright in its holder); once gripped, carried by the hand as it was when the fingers closed."""
        if grasp["off"] is None:
            return np.array(task.SCREWDRIVER_START, dtype=float), np.eye(3)
        p, r = measured_tcp("left")
        return p + r @ grasp["off"], r @ grasp["rot"]

    def take_grasp() -> None:
        p, r = measured_tcp("left")
        grasp["off"] = r.T @ (np.array(task.SCREWDRIVER_START, dtype=float) - p)
        grasp["rot"] = r.T

    def slot_point(z: float, bias: float = 0.0) -> np.ndarray:
        """The middle of the blue block's release slot (moved ``bias`` toward the cheek), z above its floor (world)."""
        o, r = block()
        x = -(tp.HOOK_SLOT_CHEEK[0] + tp.SLOT_WIDTH / 2) + bias
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
        hand = rot_z(-math.pi / 2) @ arm.down  # fingers closing across the rail (turned this way the wrist roll stays
        # inside its limit: the other way it sat at -2.09 rad and the lever pose drifted)
        yield from move(arm, g + (0, 0, 0.05), 2.5, grip=SD_OPEN, rot=hand, label="over the screwdriver")
        yield from move(arm, g, 1.2, label="down round its handle")
        yield from settle(arm, g, 0.6, "line up on the handle")
        yield from move(arm, arm.pos, 0.6, grip=0.0, label="grip the handle")
        take_grasp()
        yield from move(arm, arm.pos + (0, 0, SD_LIFT), 1.2, label="lift it out of the holder")
        yield from move(arm, arm.pos, 0.8, rot=rot_y(-SD_LEAN) @ arm.rot, label="lean it away from the block")
        rep["picked"] = bool(driver()[0][2] > tip[2] + 0.04)
        # over the release slot, the blade's tip lined up on the slot's middle by its measured position (the right
        # arm waits where it starts until the screwdriver is done: the two arms working at the block together met
        # over it, runs 11-24)
        tip, _ = driver()
        above = slot_point(SLOT_ABOVE)
        yield from move(arm, arm.pos + (above - tip) + np.asarray(SD_FRONT), 2.0, label="high over the release slot")
        yield from move(arm, arm.pos - np.asarray(SD_FRONT), 1.5, label="down over the slot")
        # into the slot, lined up by the blade tip's measured position; if the blade comes down on the slot's edge
        # instead, it is lifted and tried again a little to either side (as by hand)
        tries = []
        for bias in SLOT_TRIES:
            err = yield from tip_to(arm, slot_point(SLOT_ABOVE, bias), SD_CENTRE_TRIES, "line the blade up over the slot")
            tip, _ = driver()
            yield from move(arm, arm.pos + (slot_point(SLOT_IN, bias) - tip), 1.2, label="blade into the slot")
            yield from tip_to(arm, slot_point(SLOT_IN, bias), 2, "line up in the slot", xy_only=True)
            yield from move(arm, arm.pos, 0.4, label="settle in the slot")
            depth = float((block()[1].T @ (driver()[0] - slot_point(0.0)))[2])
            tries.append([round(bias / MM, 2), round(err / MM, 3), round(depth / MM, 2)])
            if depth < SLOT_IN_OK:
                break
            tip, _ = driver()
            yield from move(arm, arm.pos + (slot_point(SLOT_ABOVE) - tip) * np.array([0.0, 0.0, 1.0]), 0.8,
                            label="lift the blade off the slot's edge")
        rep["insert_tries"] = tries
        tip, _ = driver()
        rep["in_slot_mm"] = [round(float(v) / MM, 2) for v in (tip - slot_point(0.0))]
        # (the slot is open at its sides: while levering, the blade's tip is kept on the slot's middle along the rail
        # by its measured position, or it walks sideways off the tab)
        base_pos = arm.pos.copy()
        ycorr = 0.0

        def on_slot() -> np.ndarray:
            """The command's correction keeping the blade's tip on the slot's middle along the rail."""
            nonlocal ycorr
            tip, want = driver()[0], slot_point(SLOT_IN)
            ycorr = float(np.clip(ycorr + SLOT_Y_GAIN * (want[1] - tip[1]), -0.03, 0.03))
            return np.array([0.0, ycorr, 0.0])

        # lever: the handle is tipped toward the block about the slot's top edge on the housing's cheek (the fulcrum),
        # so the blade's tip pushes the foot's tab, and with it the foot, out toward the arms
        o, r = block()
        fulcrum = o + r @ np.array([-tp.HOOK_SLOT_CHEEK[0], 0.0, tp.SLOT_Z[1]])
        rot0 = arm.rot.copy()
        ang = 0.0

        def tooth_out() -> float:
            """How far the sprung foot has moved out relative to the rail (its tooth clears the lip at 2.0 mm)."""
            b = on_rail()
            return b["feet_spread"][1] - b["across"]

        push = 0.0

        def levered(a: float) -> None:
            rr = rot_y(a)
            arm.pos = fulcrum + rr @ (base_pos - fulcrum) + on_slot() - np.array([push, 0.0, 0.0])
            arm.rot = rr @ rot0

        phase["left"] = "lever the foot out"
        for _ in range(round(LEVER_S / DT)):  # a fixed ramp: the tilt and the push toward the arms, the command leading
            ang = min(ang + math.radians(LEVER_RATE_DEG) * DT, math.radians(LEVER_MAX_DEG))
            push = min(push + LEVER_PUSH_RATE * DT, LEVER_PUSH_MAX)
            levered(ang)
            yield
        rep["lever_push_mm"] = round(push / MM, 1)
        rep["lever_deg"], rep["foot_spread_mm"] = round(math.degrees(ang), 1), round(on_rail()["feet_spread"][1] / MM, 2)
        rep["tooth_out_mm"] = round(tooth_out() / MM, 2)  # (logged from the simulation; not used by the script)
        rep["open_step"] = world["step"]
        flags.add("foot_open")
        if variant == "lever_only":  # keep the foot levered out to the end: released, but the block left on the rail
            phase["left"] = "hold the foot levered out"
            report["script_done_step"] = world["step"] + round(3.0 / DT)
            while True:
                levered(ang)
                yield
        # with the foot's tooth out from under its lip, the preloaded seat pads lift the near end; hold the foot out
        # until it has risen (the tooth then rests against the lip's edge), then draw the blade out and move away
        # keeping the foot levered out, draw the screwdriver up: the blade, wedged between the cheek and the tab,
        # carries the near end up with it until the foot's tooth is above the lip
        phase["left"] = "lever and lift the near end"
        up = 0.0
        for _ in range(round(POP_UP / POP_RATE / DT)):
            up = min(up + POP_RATE * DT, POP_UP)
            levered(ang)
            arm.pos = arm.pos + np.array([0.0, 0.0, up])
            yield
        rep["lift_cmd_mm"] = round(up / MM, 1)
        rep["popped_step"], rep["near_lift_mm"] = world["step"], round(on_rail()["feet"][1] / MM, 2)
        if variant == "probe":  # experiment: hold the screwdriver where it is
            phase["left"] = "hold"
            report["script_done_step"] = world["step"] + round(3.0 / DT)
            while True:
                yield
        yield from move(arm, arm.pos + np.asarray(BLADE_OUT), 1.0, label="blade out of the slot")
        rep["after_blade_out"] = {"near_lift_mm": round(on_rail()["feet"][1] / MM, 2),
                                  "foot_out_mm": round(on_rail()["feet_spread"][1] / MM, 2)}
        yield from move(arm, arm.pos + (-0.04, 0.10, 0.02), 1.5, label="out of the way")
        flags.add("left_clear")
        phase["left"] = "done"
        while True:
            yield

    # -- right arm: take the blue block off, into the tray -------------------------------------------------
    def right_script():
        arm = arms["right"]
        if variant == "probe":
            phase["right"] = "idle"
            while True:
                yield
        if variant == "endgrip":  # experiment: grip the block by its two ends, the fingers closing across the rail
            yield from wait_for(arm, "left_clear", "wait for the screwdriver")
            o, _ = block()
            hand = rot_z(math.pi / 2) @ arm.down
            mid = o + np.array([ENDGRIP_X, 0.0, ENDGRIP_Z + PAD_TIP])
            yield from move(arm, mid + (0, 0, 0.10), 2.5, grip=ENDGRIP_OPEN, rot=hand, label="over the block's ends")
            yield from move(arm, mid + (0, 0, 0.04), 1.5, label="down")
            yield from move(arm, mid, 2.0, label="down beside its ends")
            report["endgrip_z_err_mm"] = round((measured_tcp("right")[0][2] - mid[2]) / MM, 1)
            yield from move(arm, arm.pos, 1.0, grip=0.0, label="close on the ends")
            yield from move(arm, arm.pos + (0, 0, 0.03), 1.5, label="lift")
            report["script_done_step"] = world["step"] + round(2.0 / DT)
            while True:
                yield
        o, r = block()
        g = o + r @ np.array([GRIP_FAR, 0.0, tp.BLOCK_HEIGHT - GRASP_DEPTH]) + np.array([0, 0, PAD_TIP])
        lean = rot_y(GRIP_LEAN) @ arm.down
        approach = lean[:, 2]
        g = o + r @ np.array([GRIP_FAR, 0.0, tp.BLOCK_HEIGHT - GRASP_DEPTH]) - PAD_TIP * approach
        # 1. once the screwdriver has released the foot and is out of the way, grip the block's far end, coming in high
        # over it (the two arms working at the block together met over it, runs 11-24)
        yield from wait_for(arm, "left_clear", "wait for the screwdriver")
        o, r = block()
        g = o + r @ np.array([GRIP_FAR, 0.0, tp.BLOCK_HEIGHT - GRASP_DEPTH]) - PAD_TIP * approach
        high = g + np.asarray(HOVER)
        high[2] = max(high[2], arm.pos[2])
        yield from move(arm, high, 2.0, grip=OPEN, rot=lean, label="high over the block's far end")
        yield from move(arm, g + (0, 0, 0.06), 1.0, label="over the blue block")
        yield from move(arm, g + (0, 0, 0.012), 1.0, label="down near it")
        yield from settle(arm, g + (0, 0, 0.012), 0.6, "line up over it")
        corr = arm.pos - (g + (0, 0, 0.012))
        yield from move(arm, g + corr, 0.6, label="down to it")
        yield from settle(arm, g, 0.5, "line up over it")
        yield from move(arm, arm.pos, 0.8, grip=0.0, label="grip its broad faces")
        p_g, r_g = measured_tcp("right")
        o0, _ = block()
        low_below_tcp = p_g[2] - (o0[2] + tp.bounds("block")[0][2])  # the block's lowest point below the TCP
        flags.add("gripped")
        # 2. once the foot is levered out, tip the near end up along an arc about the hook until it is clear of its lip
        yield from wait_for(arm, "foot_open", "wait for the foot to be levered out")
        phase["right"] = "tip the near end up"
        for _ in range(round(TIP_CMD_DEG / TIP_RATE_DEG / DT)):  # a fixed arc about the hook, the command leading
            o, r = block()
            pivot = o + r @ np.array([tp.LIP_EDGE - 2.5 * MM, 0.0, -tp.LIP_THICK])
            step_a = math.radians(TIP_RATE_DEG) * DT
            c, sn = math.cos(step_a), math.sin(step_a)
            rel = arm.pos - pivot
            arm.pos = pivot + np.array([c * rel[0] + sn * rel[2], rel[1], -sn * rel[0] + c * rel[2]])
            yield
        report["tipped_step"], report["tip_deg"] = world["step"], round(on_rail()["tilt_deg"], 1)
        report["tip_near_lift_mm"] = round(on_rail()["feet"][1] / MM, 2)
        flags.add("tipped")
        # 3. slide it toward the far side until the hook is out from under its lip, the command leading
        # (along the block's own tipped length, so the hook's tooth comes out level from under its lip; slid level
        # across the table, the tooth jammed against the lip and the block turned in the grip, run 27)
        phase["right"] = "slide toward the far side"
        start = arm.pos.copy()
        # the block's tipped length, from how far the hand holding it has turned since it gripped (joint readings)
        along = (measured_tcp("right")[1] @ r_g.T) @ np.array([1.0, 0.0, 0.0])
        lead = 0.0
        for _ in range(round(SLIDE_LEAD / SLIDE_RATE / DT)):
            lead = min(lead + SLIDE_RATE * DT, SLIDE_LEAD)
            arm.pos = start + lead * along
            yield
        report["unhooked_step"], report["slide_lead_mm"] = world["step"], round(lead / MM, 1)
        report["unhooked_mm"] = round(on_rail()["across"] / MM, 2)
        # 4. lift it clear (take the lead back out first, so it comes straight up)
        if variant == "lifted":  # partial credit: lifted clear of the rail (0.5), then held
            up = 0.06
            yield from move(arm, arm.pos - (lead - 2 * MM) * along + (0, 0, up), 1.5,
                            label=f"lift it {round(up / MM)} mm and hold")
            phase["right"] = "hold it"
            report["script_done_step"] = world["step"] + round(3.0 / DT)
            while True:
                yield
        yield from move(arm, arm.pos - (lead - 2 * MM) * along + (0, 0, 0.06), 1.5, label="lift it clear")
        report["off_step"] = world["step"]
        # to the tray
        spot = np.array([task.TRAY_ORIGIN[0], task.TRAY_ORIGIN[1], task.TOP + tp.TRAY_FLOOR])
        yield from move(arm, (spot[0], spot[1], arm.pos[2] + 0.02), 2.5, rot=arm.down, label="carry it")
        low = measured_tcp("right")[0][2] - low_below_tcp  # the block's lowest point, carried as gripped
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
        sd_tip, sd_rot = true_driver()
        row = {"sd_tip": [round(float(v), 4) for v in sd_tip], "sd_est": [round(float(v), 4) for v in driver()[0]], "sd_tilt_deg": round(math.degrees(math.acos(min(1.0, abs(sd_rot[2, 2])))), 2),
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
                      "latched": task.latched(b)}
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
