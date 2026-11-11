"""Scripted demo for the turn-the-reversed-block task, closed loop on the simulator's object poses.

    python demo.py                    # right arm takes the block off, turns it round and remounts it: 1.0
    python demo.py --variant hold     # no arm moves: the block stays reversed (0.0)
    python demo.py --device cpu       # CPU physics, ~3x faster, for quick checks

It drives the ``yam_arena`` embodiment directly (20 Hz control, object_jitter 0, as a trial runs) and reads the
task's own scoring from each step's ``info``. The script knows which block is reversed (a policy has to see it).
Right arm, all in the block's own frame (its fixed hook toward +x):

1. grips the block across its broad faces, 18 mm below its top;
2. takes it off: pushes it toward its sprung foot's end until it has moved 2.4 mm (the fixed hook gives and the
   sprung foot comes out from under its lip), tips the foot's end up 6 degrees about the hook, slides it toward the
   hook's end until the hook is out, lifts it clear;
3. turns it round: 180 degrees about the vertical;
4. hooks it: brings it over its place tipped 13 degrees (near end up) and 3.5 mm toward the far side, so the hook's
   tooth is just beside the far lip, then slides it toward the arms until the tooth is under the lip, correcting its
   position along the rail as it goes;
5. presses it home: still holding it, rotates it down about the hook, the command leading downward, until it
   latches; holds 0.5 s; lets go and lifts away.

The left arm does not move. Writes ``result.json``, ``steps.jsonl`` and (unless ``--no-video``) one video per camera
to ``--out``.
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
    "One of the terminal blocks on the DIN rail is mounted the wrong way round. Take it off and mount it the right "
    "way round in the same place, without disturbing the other blocks."
)
VARIANTS = ("full", "hold")
CAMERAS = ("top_cam", "left_cam", "right_cam")
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ
MAX_STEPS = 1200
MM = 1e-3

PAD_TIP = 0.013  # pad tips beyond the TCP
FIX_ARM, HOLD_ARM = "right", "left"  # the right arm fixes the block
USE_HOLD = False  # the rail is screwed down: no arm holds it (True: HOLD_ARM presses it, for a loose rail)
TURN_SIGN = 1  # the right arm turns the block anticlockwise seen from above (its wrist has room that way)
HOLD_AT = -0.229  # with USE_HOLD, HOLD_ARM presses the rail here (rail frame y): in the stub beyond its end bracket
HOLD_X = 8 * MM  # in the channel, this far from the centreline toward the arms
HOLD_PRESS = 40 * MM  # the fingertips commanded this far into the rail (the arm is soft: this sets the press force)
OPEN = 0.35  # gripper opening (0..1 of 94 mm): ~33 mm between the pads
GRASP_DEPTH = 18 * MM  # pad tips this far below the block's top
PUSH_RATE, PUSH_MAX = 10 * MM, 40 * MM  # the push command leads by this much more per s, at most this far
PUSH_DOWN = 5.0 * MM  # while pushing, the command also leads this far downward (so the block does not rock)
PUSHED = 2.4 * MM  # pushed this far toward the sprung foot's end: the foot's tooth is out past its lip's edge
TIP_DEG = 6.0  # then tip the foot's end up this much about the hook
TIP_RATE_DEG = 6.0  # per second
UNHOOKED = 3.5 * MM  # then slide toward the hook's end until the block is this far beyond where it started
SLIDE_RATE, SLIDE_MAX = 10 * MM, 40 * MM
PRE_TILT_DEG = 13.0  # remounting: the block tipped this much (near end up) about its hook tooth ...
PRE_SHIFT = 3.5 * MM  # ... and this far toward the far side of its place, the tooth just beside the far lip
PRE_DROP = 0.0 * MM  # the hook's tooth this far below where it sits when latched, so it passes under the lip
HOOK_DOWN = 1.5 * MM  # while sliding, the command also leads this far downward (the arm is soft)
HOOKED_TOL = 0.3 * MM  # slide until the hook's tooth is within this of where it sits when latched
HOOK_RATE = 5 * MM  # per second
POSE_TOL = 0.6 * MM
LIFT = 0.035  # lift the block this far once it is off (the strip is close to the arm's base: higher is out of reach)  # line the held block's origin up to within this
PRESS_RATE, PRESS_MAX = 20 * MM, 40 * MM
HOLD_S = 0.5  # keep pressing this long once latched
LOWER_RATE_DEG = 4.0  # rotate the held block down about its hook this fast (degrees per second)
ALONG_GAIN, ALONG_MAX = 0.1, 6 * MM  # correct the block's position along the rail, per step, at most this far
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
    obs = emb.reset(Scene(id=f"reverse-block-{variant}", instruction=INSTRUCTION), seed=0)
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

    def idle():
        while True:
            yield

    def axis_rot(axis, a):
        """Rotation by ``a`` about the unit vector ``axis`` (Rodrigues)."""
        k = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + math.sin(a) * kx + (1 - math.cos(a)) * kx @ kx

    HOOK = np.array([tp.LIP_EDGE - tp.A3C_HOOK_REACH * MM, 0.0, -tp.LIP_THICK - tp.SEATED_HEIGHT])  # hook tooth's
    # inner top corner, block frame

    def hold_block_at(arm, o_goal, r_goal, label, max_s=4.0, gain=0.3, limit=0.02):
        """Move the held block to the pose (o_goal, r_goal): the TCP pose from the grip measured when the block was
        lifted, corrected step by step by the block's measured position error, until it is within POSE_TOL."""
        phase[arm.side] = label
        r_tcp = r_goal @ grip_rel["r"].T
        p_tcp = o_goal - r_tcp @ grip_rel["t"]
        arm.rot = r_tcp
        corr, ok = arm.pos - p_tcp, 0
        corr = np.clip(corr, -limit, limit)
        for _ in range(round(max_s / DT)):
            err = o_goal - block()[0]
            ok = ok + 1 if float(np.linalg.norm(err)) <= POSE_TOL else 0
            if ok >= 4:
                break
            corr = np.clip(corr + gain * err, -limit, limit)
            arm.pos = p_tcp + corr
            yield
        report.setdefault("pose_err_mm", {})[label] = [round(float(v) / MM, 2) for v in (o_goal - block()[0])]

    grip_rel = {}

    # -- (USE_HOLD only) the other arm holds a loose rail down ----------------------------------------------
    def hold_script():
        arm = arms[HOLD_ARM]
        if variant == "hold" or not USE_HOLD:
            phase[arm.side] = "idle"
            yield from idle()
        arm.grip = 0.0
        down = rot_z(rail_yaw()) @ rot_z(math.pi / 2) @ arm.down  # closed fingertips, the pads' width along the rail
        tcp = rail_point((-HOLD_X, HOLD_AT, tp.SHEET)) - PAD_TIP * down[:, 2]  # in the channel, on its floor
        yield from move(arm, tcp + (0, 0, 0.05), 2.5, grip=0.0, rot=down, label="to the hold point")
        yield from move(arm, tcp + (0, 0, 0.012), 1.5, label="down to the rail")
        yield from move(arm, tcp - (0, 0, HOLD_PRESS), 1.0, label="press the rail down")
        flags.add("holding")
        phase[arm.side] = "holding the rail"
        yield from idle()

    # -- the fixing arm: take the reversed block off, turn it round, remount it ------------------------------
    def fix_script():
        arm = arms[FIX_ARM]
        if variant == "hold":
            phase[arm.side] = "idle"
            yield from idle()
        if USE_HOLD:
            yield from wait_for(arm, "holding", "wait for the rail to be held")
        o, r = block()
        g = o + r @ np.array([0.0, 0.0, tp.BLOCK_HEIGHT - GRASP_DEPTH]) + np.array([0, 0, PAD_TIP])
        yield from move(arm, g + (0, 0, 0.04), 2.5, grip=OPEN, rot=arm.down, label="over the block")
        yield from move(arm, g + (0, 0, 0.012), 1.2, label="down near it")
        yield from settle(arm, g + (0, 0, 0.012), 0.8, "line up over it")
        corr = arm.pos - (g + (0, 0, 0.012))
        yield from move(arm, g + corr, 0.6, label="down to it")
        yield from settle(arm, g, 0.5, "line up over it")
        yield from move(arm, arm.pos, 0.8, grip=0.0, label="grip its broad faces")
        report["grip_step"] = world["step"]
        ex, ey = r @ np.array([1.0, 0, 0]), r @ np.array([0, 1.0, 0])  # toward its hook; along the rail
        _, r_r = task.rail_pose(world["states"])
        sgn = float(np.sign((r_r.T @ ex)[0]))  # +1: its hook toward the far side
        # 1. push it toward its sprung foot's end, the command leading, until the foot is out
        base_pos, lead = arm.pos.copy(), 0.0
        phase[arm.side] = "push toward its foot"
        for _ in range(round(6.0 / DT)):
            if -sgn * on_rail()["across"] >= PUSHED:
                break
            lead = min(lead + PUSH_RATE * DT, PUSH_MAX)
            arm.pos = base_pos - ex * lead - np.array([0.0, 0.0, min(lead, PUSH_DOWN)])  # keep its hook end down
            yield
        report["pushed_step"], report["push_lead_mm"] = world["step"], round(lead / MM, 1)
        # 2. tip its foot's end up about the hook
        phase[arm.side] = "tip its foot's end up"
        for _ in range(round(4.0 / DT)):
            if on_rail()["tilt_deg"] >= TIP_DEG:
                break
            o, r = block()
            pivot = o + r @ HOOK
            arm.pos = pivot + axis_rot(r @ np.array([0, 1.0, 0]), math.radians(TIP_RATE_DEG) * DT) @ (arm.pos - pivot)
            yield
        report["tipped_step"], report["tip_deg"] = world["step"], round(on_rail()["tilt_deg"], 1)
        # 3. slide it toward its hook's end until the hook is out
        phase[arm.side] = "slide toward its hook"
        start, lead = arm.pos.copy(), 0.0
        for _ in range(round(6.0 / DT)):
            if sgn * on_rail()["across"] >= UNHOOKED:
                break
            lead = min(lead + SLIDE_RATE * DT, SLIDE_MAX)
            arm.pos = start + ex * lead
            yield
        report["unhooked_step"], report["slide_lead_mm"] = world["step"], round(lead / MM, 1)
        yield from move(arm, arm.pos - ex * (lead - 2 * MM) + np.array([0, 0, LIFT]), 1.5, label="lift it clear")
        report["off_step"] = world["step"]
        # 4. turn it round about the vertical, in two halves
        r0 = arm.rot.copy()
        yield from move(arm, arm.pos, 1.5, rot=rot_z(TURN_SIGN * math.pi / 2) @ r0, label="turn it round")
        yield from move(arm, arm.pos, 1.5, rot=rot_z(TURN_SIGN * math.pi) @ r0, label="turn it round")
        yield from move(arm, arm.pos, 0.5, label="steady")
        p_tcp, r_tcp = measured_tcp(arm.side)
        o, r = block()
        grip_rel["t"], grip_rel["r"] = r_tcp.T @ (o - p_tcp), r_tcp.T @ r
        report["turned_step"], report["turned_yaw_deg"] = world["step"], round(on_rail()["yaw_deg"], 1)
        # 5. hook it under the far lip: tipped and shifted toward the far side, then slid toward the arms
        _, r_r = task.rail_pose(world["states"])
        seat = rail_point((0.0, task.START[task.TARGET], tp.RAIL_HEIGHT + tp.SEATED_HEIGHT))
        pivot = seat + r_r @ HOOK
        r_tip = r_r @ rot_y(math.radians(PRE_TILT_DEG))
        o_pre = pivot + r_tip @ (-HOOK) + r_r @ np.array([PRE_SHIFT, 0.0, -PRE_DROP])
        yield from hold_block_at(arm, o_pre + np.array([0, 0, 0.02]), r_tip, "over its place, tipped")
        yield from hold_block_at(arm, o_pre, r_tip, "down beside the far lip")
        # slide toward the arms, the command leading a little, until the hook's tooth is under the far lip
        phase[arm.side] = "slide the hook under"
        start, lead, along = arm.pos.copy(), 0.0, 0.0
        tooth_seated = float(HOOK[0])  # the tooth's inner corner, across the rail, when latched
        place = task.START[task.TARGET]
        for _ in range(round(6.0 / DT)):
            o, r = block()
            tooth = r_r.T @ (o + r @ HOOK - task.rail_pose(world["states"])[0])
            if tooth[0] <= tooth_seated + HOOKED_TOL:
                break
            lead = min(lead + HOOK_RATE * DT, PRE_SHIFT + 10 * MM)
            along = float(np.clip(along + ALONG_GAIN * (place - on_rail()["along"]), -ALONG_MAX, ALONG_MAX))
            arm.pos = start + r_r @ np.array([-lead, along, -min(lead, HOOK_DOWN)])
            yield
        report["hook_lead_mm"] = round(lead / MM, 1)
        report["tooth_mm"] = [round(float(v) / MM, 2) for v in tooth]
        report["hooked_step"] = world["step"]
        # 6. still holding it, rotate it down about its hook until the sprung foot snaps under the near lip
        phase[arm.side] = "lower it home"
        held, down = 0, 0.0
        for _ in range(round(10.0 / DT)):
            if task.stage(world["states"]) in ("fixed", "elsewhere"):
                report.setdefault("latched_step", world["step"])
                held += 1
                if held >= round(HOLD_S / DT):
                    break
            o, r = block()
            pivot = o + r @ HOOK
            if on_rail()["tilt_deg"] > 0.5:  # rotate the command about the hook (the near end down) ...
                rot = axis_rot(r @ np.array([0, 1.0, 0]), -math.radians(LOWER_RATE_DEG) * DT)
                arm.pos = pivot + rot @ (arm.pos - pivot)
                arm.rot = rot @ arm.rot
            down = min(down + PRESS_RATE * DT, PRESS_MAX)  # ... and lead it downward (the arm is soft)
            along = float(np.clip(along + ALONG_GAIN * (place - on_rail()["along"]), -ALONG_MAX, ALONG_MAX))
            arm.pos = arm.pos + r_r @ np.array([0.0, ALONG_GAIN * (place - on_rail()["along"]), 0.0]) \
                - np.array([0.0, 0.0, PRESS_RATE * DT if down < PRESS_MAX else 0.0])
            yield
        report["press_lead_mm"] = round(down / MM, 1)
        report["latched_state"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in on_rail().items()
                                   if k in ("across", "along", "height", "tilt_deg", "yaw_deg")}
        arm.pos = measured_tcp(arm.side)[0].copy()  # take the downward lead out, so letting go does not push on
        yield from move(arm, arm.pos, 0.8, grip=OPEN, label="let go")
        yield from move(arm, arm.pos + np.array([0, 0, 0.035]), 1.2, label="lift away")
        phase[arm.side] = "done"
        report["script_done_step"] = world["step"]
        yield from idle()

    scripts = {FIX_ARM: fix_script(), HOLD_ARM: hold_script()}
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
        row = {"step": step, "left": phase["left"], "right": phase["right"], "score": res.info.get("score"),
               "success": res.info.get("success"), "stage": task.stage(st), "undisturbed": task.undisturbed(st)}
        for n in task.BLOCKS:
            b = on_rail(n)
            row[n] = {"along_mm": round(b["along"] / MM, 2), "across_mm": round(b["across"] / MM, 2),
                      "h_mm": round(b["height"] / MM, 2), "feet_mm": [round(f / MM, 2) for f in b["feet"]],
                      "spread_mm": [round(t / MM, 2) for t in b["feet_spread"]], "tilt": round(b["tilt_deg"], 2), "yaw": round(b["yaw_deg"], 1),
                      "pads_mm": [round(t / MM, 2) for t in b["pads_squeeze"]], "latched": task.latched(b)}
        for side in ("left", "right"):
            row[f"{side[0]}_cmd"] = [round(float(v), 4) for v in arms[side].pos]
            row[f"{side[0]}_meas"] = [round(float(v), 4) for v in measured_tcp(side)[0]]
        row["fix_ik_err"] = [round(float(v), 4) for v in getattr(arms[FIX_ARM], "ik_err", (0.0, 0.0))]
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
    st = world["states"]
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
                                                               "final_stage", "undisturbed",
                                                               "wall_s")}))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
