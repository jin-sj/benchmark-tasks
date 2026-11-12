"""Scripted two-arm demo for the handover task, closed loop on the simulator's object poses.

    python demo.py --variant full      # left arm holds the rail; right arm takes the block from the hand, presses: 1.0
    python demo.py --variant no_press  # taken cleanly, set on the rail, never pressed: 0.5
    python demo.py --variant drop      # taken cleanly, then dropped on the table: capped at 0.25
    python demo.py --variant snatch    # grips and pulls away at once, before the hand lets go: 0
    python demo.py --variant rough     # presses down on the person's forearm first (over 10 N): 0
    python demo.py --variant push      # grips, then shoves the block into the hand before it lets go (over 5 N): 0
    python demo.py --variant one_arm   # the full right-arm script, the rail not held: 1.0 (the hold is optional)

It drives the ``yam_arena`` embodiment directly (20 Hz control, object_jitter 0, as a trial runs) and reads the
task's own scoring from each step's ``info``. Each arm follows Cartesian targets turned into joint targets by
yam_arena's IK; the right arm corrects them every step from the measured block and rail poses:

- left arm: fingers closed, pressed into the rail's channel in the free stub beyond the left end bracket, for the
  whole run;
- right arm: waits for the hand to hold the block out, comes down over the block's free (near) half following it as
  the hand drifts, grips it across its broad faces, holds still until the hand has let go and is withdrawing, lifts
  it back and up clear of the hand, carries it over the rail 36 mm short of the beige block, lowers it until its feet
  touch the lips, presses it straight down (still holding it) until it bottoms on the lips, lets go and moves away.

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
    "Take the terminal block from the person's hand: grip it while they still hold it, so they can let go. Then press "
    "it straight down onto the DIN rail until it clicks, without knocking the rail out of place."
)
VARIANTS = ("full", "no_press", "drop", "snatch", "rough", "push", "one_arm")
CAMERAS = ("top_cam", "left_cam", "right_cam")
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ
MAX_STEPS = 1200
MM = 1e-3

PICK = "block"
SNAP_SHORT = 36 * MM  # the block is snapped on this far short of the beige block
OPEN = 0.35  # gripper opening for the grasp (0..1 of 94 mm): ~33 mm between the pads
PAD_TIP = 0.013  # pad tips beyond the TCP
GRASP_DEPTH = 20 * MM  # pad tips this far below the block's top (2.5 mm above the person's fingertip pads)
GRASP_BACK = 0.0  # grip the block this far from its centre, toward the robot: centred, so the press is even
HOLD_PRESS = 8 * MM  # the left fingertips are commanded this far into the rail's web
PRESS_MAX = 20 * MM  # the press command goes at most this far below the touch height
PRESS_SPEED = 5 * MM  # per second
FINISH_X = 12 * MM  # finishing press: this far from the block's middle, over the side that has not clicked
FINISH_MAX = 30 * MM  # how far below the first touch height the finishing press may go
SERVO_UNTIL = 1.5 * MM  # pressing: stop correcting across the rail once the base is this close to the lips
BOTTOMED = 0.3 * MM  # pressing: the base this close to the lips at both edges ends the press
HOLD_YAW = -math.pi / 2  # left arm: fingers closing across the rail
HOLD_LEAN = -math.radians(34.0)  # left arm: tips toward the block, wrist away from the right arm
AFTER_SUCCESS_S = 3.0  # the demo keeps running this long after success, hands off
CARRY_HEIGHT = 0.11  # TCP height above the table while carrying the block to the rail
ROUGH_PRESS = 50 * MM  # "rough": the closed fingertips are commanded this far into the forearm
PUSH_INTO = 55 * MM  # "push": the gripped block is shoved this far toward the palm while the hand still holds it


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
    obs = emb.reset(Scene(id=f"handover-{variant}", instruction=INSTRUCTION), seed=0)
    set_rendering(emb, video)
    task = emb.task.module
    tp = task.tp
    from yam_arena.scene import object_states  # patched by the task: the jointed block and the scripted hand

    left_q, right_q = split(np.asarray(obs.state["joint_pos"]))
    arms = {
        "left": Arm("left", station.BASES["left"], left_q[:6], ik, grip_rotation),
        "right": Arm("right", station.BASES["right"], right_q[:6], ik, grip_rotation),
    }
    world = {"states": object_states(emb._env, emb.task.names), "obs": obs, "step": 0}

    def rail():
        return task.rail_pose(world["states"])

    def rail_point(local) -> np.ndarray:
        p, r = rail()
        return p + r @ np.asarray(local, dtype=float)

    def rail_yaw() -> float:
        _, r = rail()
        return math.atan2(r[1, 0], r[0, 0])

    def block_base() -> np.ndarray:
        p, _ = task._origin(world["states"][PICK], task.C_BLOCK)
        return p

    def hand():
        return world["states"]["hand"]

    def measured_tcp(side: str):
        left, right = split(np.asarray(world["obs"].state["joint_pos"]))
        q = (left if side == "left" else right)[:6]
        return ik.tcp_pose(q, arms[side].base)

    phase = {"left": "", "right": ""}

    def move(arm: Arm, pos, duration: float, grip=None, rot=None, label=""):
        """Smoothly to ``pos`` (and ``rot``, ``grip``) over ``duration`` seconds."""
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

    def wait(arm: Arm, duration: float, label=""):
        phase[arm.side] = label
        for _ in range(max(1, round(duration / DT))):
            yield

    # -- left arm: hold the rail down ---------------------------------------------------------------------
    def left_script():
        arm = arms["left"]
        if variant == "one_arm":
            phase["left"] = "out of the way"
            while True:
                yield
        arm.grip = 0.0
        down = rot_z(rail_yaw()) @ rot_x(HOLD_LEAN) @ rot_z(HOLD_YAW) @ arm.down
        hold_top = rail_point((0.0, tp.HOLD_Y, tp.SHEET))
        tcp = hold_top - PAD_TIP * down[:, 2]  # the TCP that puts the pad tips on the web
        yield from move(arm, tcp + (0, 0, 0.06), 2.5, grip=0.0, rot=down, label="to the hold point")
        yield from move(arm, tcp + (0, 0, 0.012), 1.5, label="down to the rail")
        yield from move(arm, tcp - (0, 0, HOLD_PRESS), 1.0, label="press the rail down")
        phase["left"] = "holding the rail"
        while True:
            yield

    # -- right arm: take the block from the hand, press it on -------------------------------------------------
    report = {}

    def servo(arm: Arm, desired_base, corr, gain=0.35, limit=0.02, xy_only=False):
        """Integrate the block's measured position error into the TCP correction ``corr`` (in place)."""
        err = np.asarray(desired_base) - block_base()
        if xy_only:
            err[2] = 0.0
        corr += gain * err
        n = np.linalg.norm(corr)
        if n > limit:
            corr *= limit / n

    def grasp_point() -> np.ndarray:
        """The TCP that grips the block's near half across its broad faces, pad tips GRASP_DEPTH below its top."""
        b = block_base()
        _, rot = task._origin(world["states"][PICK], task.C_BLOCK)
        p = b + rot @ np.array([-GRASP_BACK, 0.0, 0.0])
        return np.array([p[0], p[1], b[2] + tp.BLOCK_HEIGHT - GRASP_DEPTH + PAD_TIP])

    def follow(arm: Arm, dz: float, duration: float, label: str, grip=None):
        """Track the (drifting) block from above for ``duration`` s, descending to ``dz`` above the grasp point."""
        phase[arm.side] = label
        p0, g0 = arm.pos.copy(), arm.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration / DT))
        for i in range(1, n + 1):
            s = smooth(i / n)
            arm.pos = p0 + s * (grasp_point() + (0, 0, dz) - p0)
            arm.grip = g0 + s * (g1 - g0)
            yield

    def right_script():
        arm = arms["right"]
        phase["right"] = "wait for the offer"
        while hand().phase == "reach":
            yield
        if variant == "rough":
            # press down on the forearm with closed fingertips: the hand gives (1.5 N/mm), and the force exceeds 10 N
            fore = np.asarray(hand().pos) + np.array([0.09, 0.0, 0.0])
            top = fore[2] + 0.020 + PAD_TIP
            yield from move(arm, (fore[0], fore[1], top + 0.05), 2.0, grip=0.0, rot=arm.down, label="over the forearm")
            yield from move(arm, (fore[0], fore[1], top - ROUGH_PRESS), 1.5, label="press on the forearm")
            yield from move(arm, (fore[0], fore[1], top + 0.06), 1.0, label="off the forearm")
        yield from move(arm, grasp_point() + (0, 0, 0.06), 2.5, grip=OPEN, rot=arm.down, label="over the block")
        yield from follow(arm, 0.0, 1.5, "down to the block")
        if variant == "snatch":
            # close fast and pull at once: the block leaves the hand before the hand has felt a firm grip for 0.3 s
            yield from follow(arm, 0.0, 0.15, "grip fast", grip=0.0)
            report["gripped_step"] = world["step"]
            yield from move(arm, arm.pos + (-0.06, 0.0, 0.03), 0.3, label="pull it away at once")
        elif variant == "push":
            # grip, then shove the block toward the palm while the hand still holds it: no robot link touches the
            # hand, but the block drives into the palm and the pinch, and the hand is pushed back on its springs
            yield from follow(arm, 0.0, 0.25, "grip", grip=0.0)
            report["gripped_step"] = world["step"]
            yield from move(arm, arm.pos + (PUSH_INTO, 0.0, 0.0), 0.3, label="shove it into the hand")
            yield from move(arm, arm.pos, 0.5, label="keep shoving")
            report["pushed_step"] = world["step"]
            phase["right"] = "hold still until the hand lets go"
            for _ in range(round(4.0 / DT)):
                if hand().phase == "withdraw":
                    break
                yield
            yield from wait(arm, 0.3, "let the hand withdraw")
        else:
            yield from follow(arm, 0.0, 0.8, "grip across the broad faces", grip=0.0)
            report["gripped_step"] = world["step"]
            phase["right"] = "hold still until the hand lets go"
            for _ in range(round(4.0 / DT)):
                if hand().phase == "withdraw":
                    break
                yield
            report["hand_let_go_step"] = world["step"]
            yield from wait(arm, 0.3, "let the hand withdraw")
        yield from move(arm, arm.pos + (-0.05, 0.0, 0.02), 1.0, label="back, clear of the hand")
        # down to carrying height first: the arm cannot reach far toward the middle of the table this high up
        yield from move(arm, (arm.pos[0], arm.pos[1], task.TOP + CARRY_HEIGHT), 1.2, label="down to carrying height")
        report["took_step"] = world["step"]

        if variant == "drop":
            yield from move(arm, arm.pos + (-0.03, -0.03, 0.0), 0.8, label="over the table")
            yield from move(arm, arm.pos, 0.5, grip=OPEN, label="drop it")
            yield from move(arm, arm.pos + (0, 0, 0.05), 1.0, label="up")
            phase["right"] = "done"
            report["script_done_step"] = world["step"]
            while True:
                yield

        # carry over the rail, SNAP_SHORT short of the beige block
        snap_y = tp.SLOT_Y - SNAP_SHORT
        down = rot_z(rail_yaw()) @ arm.down

        def tcp_for(base_local):
            """TCP target that puts the block's base at ``base_local`` (rail frame), from the grip's measured
            offset between the commanded TCP and the block."""
            offset = arm.pos - block_base()
            return rail_point(base_local) + offset

        above = (0.0, snap_y, tp.RAIL_HEIGHT + 0.03)
        p = tcp_for(above)
        yield from move(arm, (p[0], p[1], arm.pos[2]), 2.5, rot=down, label="carry over the rail")
        yield from move(arm, tcp_for(above), 1.0, label="down over the rail")

        # align on the measured block pose, then lower until the feet touch the lips
        corr = np.zeros(3)
        nominal = arm.pos.copy()
        phase["right"] = "align"
        for _ in range(round(1.0 / DT)):
            servo(arm, rail_point(above), corr)
            arm.pos = nominal + corr
            yield
        touch = (0.0, snap_y, tp.RAIL_HEIGHT + tp.RESTING_BASE_HEIGHT + 1.0 * MM)
        start, n = rail_point(above), round(1.5 / DT)
        phase["right"] = "lower onto the lips"
        for i in range(1, n + 1):
            want = start + smooth(i / n) * (rail_point(touch) - start)
            servo(arm, want, corr)
            arm.pos = nominal + (want - start) + corr
            yield
        nominal = arm.pos - corr

        if variant == "no_press":
            rest = (0.0, snap_y, tp.RAIL_HEIGHT + tp.RESTING_BASE_HEIGHT)
            phase["right"] = "set it on the lips"
            for _ in range(round(1.0 / DT)):
                servo(arm, rail_point(rest), corr, xy_only=False)
                arm.pos = nominal + corr
                yield
        else:
            # press straight down, still holding it, correcting across the rail, until it bottoms on the lips
            phase["right"] = "press down"
            depth, t_down = 0.0, None
            for i in range(round(6.0 / DT)):
                bs = task.block_on_rail(world["states"])
                if t_down is None and max(bs["feet"]) <= BOTTOMED:
                    t_down = i
                    report["bottomed_step"] = world["step"]
                if t_down is not None and i - t_down >= round(0.3 / DT):
                    break
                if t_down is None:
                    depth = min(depth + PRESS_SPEED * DT, PRESS_MAX)
                want = rail_point((0.0, snap_y, tp.RAIL_HEIGHT + tp.RESTING_BASE_HEIGHT + 1.0 * MM - depth))
                if min(bs["feet"]) > SERVO_UNTIL:
                    servo(arm, want, corr, xy_only=True)
                arm.pos = nominal + corr - np.array([0.0, 0.0, depth])
                yield
            report["press_depth_mm"] = depth / MM
        yield from move(arm, arm.pos, 0.6, grip=OPEN, label="let go")
        yield from move(arm, arm.pos + (0, 0, 0.04), 1.0, label="up")
        if variant != "no_press":
            # finishing press: a block that rolled in the grip can end clicked on one side only; press its top down
            # with closed fingertips over the side that has not clicked, as a person would with a thumb
            for attempt in range(2):
                bs = task.block_on_rail(world["states"])
                if task.latched(bs):
                    break
                side = 1.0 if bs["feet"][0] > bs["feet"][1] else -1.0  # feet[0] is the +x edge
                report.setdefault("finishing_press", []).append({"step": world["step"], "side": side})
                b0 = block_base()
                _, rot = task._origin(world["states"][PICK], task.C_BLOCK)
                spot = b0 + rot @ np.array([side * FINISH_X, 0.0, 0.0])
                top = float(world["states"][PICK].high[2])
                above = np.array([spot[0], spot[1], top + PAD_TIP + 0.008])
                yield from move(arm, above, 1.2, grip=0.0, rot=down, label="over the unclicked side")
                phase["right"] = "finishing press"
                start = arm.pos.copy()
                for i in range(round(FINISH_MAX / PRESS_SPEED / DT)):
                    bs = task.block_on_rail(world["states"])
                    if max(bs["feet"]) <= BOTTOMED and max(bs["feet_spread"]) <= 0.5 * MM:
                        break
                    arm.pos = start - np.array([0.0, 0.0, min(PRESS_SPEED * DT * (i + 1), FINISH_MAX)])
                    yield
                yield from move(arm, arm.pos + (0, 0, 0.04), 0.8, label="up")
        report["withdrawn_step"] = world["step"]
        yield from move(arm, arm.pos + (-0.05, -0.08, 0.04), 2.0, label="away")
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
        bs = task.block_on_rail(st)
        h = st["hand"]
        shift, turn = task.rail_motion(st)
        log.append({
            "step": step, "left": phase["left"], "right": phase["right"], "score": res.info.get("score"),
            "success": res.info.get("success"), "stage": task.stage(st), "hand": h.phase,
            "released": h.released, "snatched": h.snatched, "dropped": h.dropped,
            "contact_n": round(h.contact_n, 3), "peak_contact_n": round(h.peak_contact_n, 3),
            "load_n": round(h.load_n, 3), "peak_load_n": round(h.peak_load_n, 3),
            "grip_n": [round(g, 3) for g in h.grip_n],
            "rail_shift_mm": shift / MM, "rail_turn_deg": turn, "across_mm": bs["across"] / MM,
            "along_mm": bs["along"] / MM, "feet_mm": [f / MM for f in bs["feet"]],
            "feet_spread_mm": [t / MM for t in bs["feet_spread"]], "pads_mm": [t / MM for t in bs["pads_squeeze"]],
            "yaw_deg": bs["yaw_deg"], "tilt_deg": bs["tilt_deg"], "block": [round(float(v), 4) for v in st[PICK].pos],
            "speed_mm_s": float(np.linalg.norm(st[PICK].lin_vel)) / MM,
            "r_cmd": [round(float(v), 4) for v in arms["right"].pos],
            "r_meas": [round(float(v), 4) for v in measured_tcp("right")[0]],
            "r_ik_mm": round(arms["right"].ik_err[0] / MM, 2),
            "l_cmd": [round(float(v), 4) for v in arms["left"].pos],
            "l_meas": [round(float(v), 4) for v in measured_tcp("left")[0]],
        })
        if video:
            for c in CAMERAS:
                frames[c].append(res.observation.images[c])
        done_script = "script_done_step" in report and step - report["script_done_step"] >= round(3.0 / DT)
        # outcomes that settle the score for good: run 1 s more and stop
        for key, bad in (("rail_moved_step", not task.rail_ok(st)), ("unsafe_step", not task.safe(h)),
                         ("snatched_step", h.snatched), ("dropped_step", h.dropped)):
            if key not in report and bad:
                report[key] = step
        settled = [report[k] for k in ("rail_moved_step", "unsafe_step", "snatched_step") if k in report]
        done_script = done_script or bool(settled and step - min(settled) >= round(1.0 / DT))
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
    h = st["hand"]
    summary = {
        "variant": variant,
        "task_score": 1.0 if success else float(result.info.get("score", 0.0)),
        "terminated": bool(result.terminated),
        "termination_reason": result.termination_reason,
        "steps": len(log),
        "run_limit_steps": MAX_STEPS,
        "final_info": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in result.info.items()},
        "final_stage": task.stage(st),
        "hand": {"released": h.released, "snatched": h.snatched, "dropped": h.dropped,
                 "peak_contact_n": h.peak_contact_n, "peak_load_n": h.peak_load_n},
        "rail_shift_mm": shift / MM,
        "rail_turn_deg": turn,
        "block": {k: (v / MM if k in ("across", "along") else v) for k, v in task.block_on_rail(st).items()},
        "max_rail_shift_mm": max(r["rail_shift_mm"] for r in log),
        "report": report,
        "wall_s": time.time() - t0,
    }
    summary["block"]["feet_mm"] = [f / MM for f in summary["block"].pop("feet")]
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
                                                               "final_stage", "rail_shift_mm", "hand",
                                                               "wall_s")}))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
