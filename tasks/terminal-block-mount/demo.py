"""Scripted two-arm demo for the terminal-block task, closed loop on the simulator's object poses.

    python demo.py --variant full      # left arm holds the rail; right arm picks, presses, slides: 1.0
    python demo.py --variant one_arm   # the same right-arm script, the rail not held: loses credit
    python demo.py --variant no_press  # the block is set on the rail but never pressed: 0.5
    python demo.py --variant no_slide  # pressed on and latched, never slid against the beige block: 0.75

It drives the ``yam_arena`` embodiment directly (20 Hz control, object_jitter 0, as a trial runs) and reads the
task's own scoring from each step's ``info``. Each arm follows Cartesian targets turned into joint targets by
yam_arena's IK; the right arm corrects them every step from the measured block and rail poses:

- left arm: fingers closed, pressed flat into the rail's channel in the free stub beyond the left end bracket,
  36 mm from the mounted beige block, for the whole run;
- right arm: grips the nearest blue block across its broad faces, lifts it out of the tray, carries it over the
  rail 36 mm short of the beige block, aligns it on its measured pose, lowers it until its feet touch the lips,
  presses it straight down (still holding it, correcting across the rail) until both feet click under the lips,
  lets go, then pushes its side low down with closed fingertips (tips 4 mm above the lips), the gripper leaned back
  28 degrees so the fingers touch the block only near their tips, at 8 mm/s and 3 mm/s for the last 3 mm, until the gap is under 0.3 mm at every check point;
  then at once backs the fingertips off 8 mm, lifts them clear and moves the hand away. The run continues 3 s
  after success, hands off, to check that the block stays flush.

Writes ``result.json``, ``steps.jsonl`` and (unless ``--no-video``) one H.264 video per camera to ``--out``.
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
    "Hold the DIN rail down with one arm. With the other, take a blue terminal block from the tray, snap it onto "
    "the rail by pressing it straight down, and slide it along the rail until it sits flush against the beige block."
)
VARIANTS = ("full", "one_arm", "no_press", "no_slide")
CAMERAS = ("top_cam", "left_cam", "right_cam")
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ
MAX_STEPS = 1200
MM = 1e-3

PICK = "blue_3"  # the tray's block nearest the rail
SNAP_SHORT = 36 * MM  # the block is snapped on this far short of the beige block, then slid
OPEN = 0.35  # gripper opening for the pick (0..1 of 94 mm): ~33 mm between the pads
PAD_TIP = 0.013  # pad tips beyond the TCP
GRASP_DEPTH = 20 * MM  # pad tips this far below the block's top
HOLD_PRESS = 8 * MM  # the left fingertips are commanded this far into the rail's web
PRESS_MAX = 20 * MM  # the press command goes at most this far below the touch height
SERVO_UNTIL = 1.5 * MM  # pressing: stop correcting across the rail once the base is this close to the lips
BOTTOMED = 0.3 * MM  # pressing: the base this close to the lips at both edges ends the press
LEAN = math.radians(28.0)  # sliding: the gripper leaned back so the fingers touch the block only 6-9 mm above their
# lowest point. The closed fingers widen upward: leaned 20 degrees their front is near vertical but overhangs ~1 mm at
# 40 mm, so they pushed the block's top and pitched it 3.5 degrees, jamming it on the lips
PUSH_YAW = math.pi / 2  # sliding: fingers closing across the rail, so the gripper is narrow along it
PAD_HALF = 7.0 * MM  # half the closed pads' width at the tips (along the rail when pushing)
HOLD_YAW = -math.pi / 2  # left arm: fingers closing across the rail
HOLD_LEAN = -math.radians(34.0)  # left arm: tips toward the block, wrist away from the right arm
PUSH_HEIGHT = 4.0 * MM  # the TCP's tip point above the lips while pushing (leaned, the fingers reach 2.5 mm lower)
PUSH_FAST, PUSH_SLOW, SLOW_ZONE = 8 * MM, 3 * MM, 3 * MM  # per second; slow for the last 3 mm
DONE_GAP = 0.3 * MM
BACK_OFF = 8 * MM  # after the push, the fingertips back off this far before lifting
AFTER_SUCCESS_S = 3.0  # the demo keeps running this long after success, hands off, to check the block stays flush
LEAD_TRAVEL, LEAD_SEAT = 15 * MM, 16 * MM  # how far the push command may lead the measured fingertips: sliding, and
# seating it in the last 2 mm (this sets the push force)
SEAT_ZONE = 2 * MM
SQUARE_REACH = 15 * MM  # how far off the block's middle the push may move to square it
SQUARE_GAIN_SLIDE, SQUARE_GAIN_SEAT = 4.0, 10.0  # push this far off the middle per mm the two ends' gaps differ


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
    obs = emb.reset(Scene(id=f"terminal-block-{variant}", instruction=INSTRUCTION), seed=0)
    set_rendering(emb, video)
    task = emb.task.module
    tp = task.tp
    from yam_arena.scene import object_states  # patched by the task to cover the jointed blocks

    left_q, right_q = split(np.asarray(obs.state["joint_pos"]))
    arms = {
        "left": Arm("left", station.BASES["left"], left_q[:6], ik, grip_rotation),
        "right": Arm("right", station.BASES["right"], right_q[:6], ik, grip_rotation),
    }
    world = {"states": object_states(emb._env, emb.task.names), "obs": obs}

    def rail():
        return task.rail_pose(world["states"])

    def rail_point(local) -> np.ndarray:
        p, r = rail()
        return p + r @ np.asarray(local, dtype=float)

    def rail_yaw() -> float:
        _, r = rail()
        return math.atan2(r[1, 0], r[0, 0])

    def block_base(name=PICK) -> np.ndarray:
        p, _ = task._origin(world["states"][name], task.C_BLOCK)
        return p

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

    # -- right arm: pick, press, slide --------------------------------------------------------------------
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

    def right_script():
        arm = arms["right"]
        b = block_base()
        top = b[2] + tp.BLOCK_HEIGHT
        grasp = np.array([b[0], b[1], top - GRASP_DEPTH + PAD_TIP])
        yield from wait(arm, 0.5, "start")
        yield from move(arm, grasp + (0, 0, 0.06), 2.5, grip=OPEN, rot=arm.down, label="over the block")
        yield from move(arm, grasp, 1.5, label="down to the block")
        yield from move(arm, grasp, 0.8, grip=0.0, label="grip across the broad faces")
        yield from move(arm, grasp + (0, 0, 0.07), 1.2, label="lift")
        report["lifted"] = True

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
        yield from move(arm, (p[0], p[1], arm.pos[2]), 2.0, rot=down, label="carry over the rail")
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
            # press straight down, still holding it, correcting across the rail, until both feet click under
            phase["right"] = "press down"
            depth, t_down = 0.0, None
            for i in range(round(6.0 / DT)):
                bs = task.block_on_rail(world["states"], PICK)
                # stop going deeper once the base is down on the lips at both edges: pressing on would only crush a
                # block the hand holds slightly off-centre; once let go, the feet's arms centre it
                if t_down is None and max(bs["feet"]) <= BOTTOMED:
                    t_down = i
                    report["bottomed_step"] = world["step"]
                if t_down is not None and i - t_down >= round(0.3 / DT):
                    break
                if t_down is None:
                    depth = min(depth + 5 * MM * DT, PRESS_MAX)
                want = rail_point((0.0, snap_y, tp.RAIL_HEIGHT + tp.RESTING_BASE_HEIGHT + 1.0 * MM - depth))
                # correct across the rail only until the feet reach the lip edges: from there the lead-ins centre
                # the block, and a correction still winding up would shove a half-clicked block sideways
                if min(bs["feet"]) > SERVO_UNTIL:
                    servo(arm, want, corr, xy_only=True)
                arm.pos = nominal + corr - np.array([0.0, 0.0, depth])
                yield
            report["press_depth_mm"] = depth / MM
        yield from move(arm, arm.pos, 0.6, grip=OPEN, label="let go")
        yield from move(arm, arm.pos + (0, 0, 0.04), 1.0, label="up")
        if variant in ("no_press", "no_slide"):
            yield from move(arm, arm.pos + (-0.05, -0.08, 0.04), 2.0, label="away")
            phase["right"] = "done"
            report["script_done_step"] = world["step"]
            while True:
                yield
            return

        # slide: push the block's side with closed fingertips, the gripper leaned back so the tips lead
        lean = rot_z(rail_yaw()) @ rot_x(LEAN) @ rot_z(PUSH_YAW) @ arm.down
        approach = lean[:, 2]

        def tcp_for_tip(tip_local):
            return rail_point(tip_local) - PAD_TIP * approach

        bs = task.block_on_rail(world["states"], PICK)
        tip_y = bs["along"] - tp.BLOCK_WIDTH / 2 - PAD_HALF - 4 * MM
        tip_z = tp.RAIL_HEIGHT + PUSH_HEIGHT
        yield from move(arm, tcp_for_tip((bs["across"], tip_y, tip_z + 0.03)), 1.5, grip=0.0, rot=lean,
                        label="behind the block")
        yield from move(arm, tcp_for_tip((bs["across"], tip_y, tip_z)), 1.0, label="down beside the block")
        phase["right"] = "push"
        pushing_from = world["step"]
        while True:
            bs = task.block_on_rail(world["states"], PICK)
            gap = max(bs["gaps"])
            if all(g <= DONE_GAP for g in bs["gaps"]):
                break
            speed = PUSH_SLOW if gap < SLOW_ZONE else PUSH_FAST
            # don't run the command more than a few mm ahead of where the fingertips really are
            tcp, rot = measured_tcp("right")
            tip_real = task.rail_pose(world["states"])[1].T @ (tcp + PAD_TIP * rot[:, 2] - rail()[0])
            tip_y = min(tip_y + speed * DT, tip_real[1] + (LEAD_SEAT if gap < SEAT_ZONE else LEAD_TRAVEL))
            # push nearer the end that lags, so the block squares up against the beige one
            g_p, g_n = max(bs["gaps"][:2]), max(bs["gaps"][2:])
            # (all the way: a yaw left to grow while sliding swings the block flush in one hard knock at the end)
            gain = SQUARE_GAIN_SEAT if gap < SEAT_ZONE else SQUARE_GAIN_SLIDE
            push_x = bs["across"] + float(np.clip((g_p - g_n) * gain, -SQUARE_REACH, SQUARE_REACH))
            arm.pos = tcp_for_tip((push_x, tip_y, tip_z))
            yield
        report["push_steps"] = world["step"] - pushing_from
        report["flush_step"] = world["step"]
        # let go at once: back the fingertips off the block, lift them clear and move the hand away
        tcp, rot = measured_tcp("right")
        tip_real = task.rail_pose(world["states"])[1].T @ (tcp + PAD_TIP * rot[:, 2] - rail()[0])
        yield from move(arm, tcp_for_tip((push_x, tip_real[1] - BACK_OFF, tip_z)), 0.3, label="back off")
        yield from move(arm, arm.pos + (0, 0, 0.04), 0.8, label="lift clear")
        report["withdrawn_step"] = world["step"]
        yield from move(arm, arm.pos + (-0.05, -0.06, 0.02), 1.5, label="away")
        phase["right"] = "done"
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
        bs = task.block_on_rail(st, PICK)
        shift, turn = task.rail_motion(st)
        log.append({
            "step": step, "left": phase["left"], "right": phase["right"], "score": res.info.get("score"),
            "success": res.info.get("success"), "stage": task.stage(st, PICK), "rail_shift_mm": shift / MM,
            "rail_turn_deg": turn, "across_mm": bs["across"] / MM, "along_mm": bs["along"] / MM,
            "feet_mm": [f / MM for f in bs["feet"]],
            "feet_spread_mm": [t / MM for t in bs["feet_spread"]],
            "pads_mm": [t / MM for t in bs["pads_squeeze"]], "gaps_mm": [g / MM for g in bs["gaps"]],
            "yaw_deg": bs["yaw_deg"], "tilt_deg": bs["tilt_deg"],
            "speed_mm_s": float(np.linalg.norm(st[PICK].lin_vel)) / MM, "spin_rad_s": float(np.linalg.norm(st[PICK].ang_vel)),
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
        # the rail knocked out of place caps the score for good: note it, run 1 s more and stop
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
    if success_step is not None:
        after = log[success_step - 1:]
        hands_off = [r for r in log if "withdrawn_step" in report and r["step"] >= report["withdrawn_step"]]
        report["after_success"] = {
            "seconds": (len(after) - 1) * DT,
            "stage_at_end": after[-1]["stage"],
            "gaps_at_success_mm": [round(g, 3) for g in after[0]["gaps_mm"]],
            "gaps_at_end_mm": [round(g, 3) for g in after[-1]["gaps_mm"]],
            "max_gap_after_mm": round(max(max(r["gaps_mm"]) for r in after), 3),
            "hands_off_steps_before_success": max(0, success_step - report.get("withdrawn_step", success_step)),
            "flush_and_hands_off_seconds": round(sum(1 for r in hands_off if r["stage"] == "flush") * DT, 2),
        }
    summary = {
        "variant": variant,
        "task_score": 1.0 if success else float(result.info.get("score", 0.0)),
        "terminated": bool(result.terminated),
        "termination_reason": result.termination_reason,
        "steps": len(log),
        "run_limit_steps": MAX_STEPS,
        "final_info": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in result.info.items()},
        "final_stage": task.stage(st, PICK),
        "rail_shift_mm": shift / MM,
        "rail_turn_deg": turn,
        "block": {k: (v / MM if k in ("across", "along") else v) for k, v in task.block_on_rail(st, PICK).items()},
        "max_rail_shift_mm": max(r["rail_shift_mm"] for r in log),
        "max_rail_turn_deg": max(r["rail_turn_deg"] for r in log),
        "report": report,
        "wall_s": time.time() - t0,
    }
    summary["block"]["feet_mm"] = [f / MM for f in summary["block"].pop("feet")]
    summary["block"]["gaps_mm"] = [g / MM for g in summary["block"].pop("gaps")]
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(summary, indent=2))
    with open(out / "steps.jsonl", "w") as f:
        for row in log:
            f.write(json.dumps(row) + "\n")
    if video:
        write_videos(frames, out, variant)
        import cv2

        first = np.ascontiguousarray(frames["top_cam"][0][..., ::-1])
        cv2.imwrite(str(out / "setup.png"), first)
        # the parts are small in the overhead view: also a 3x close-up of the table centre
        crop = first[140:280, 220:470]
        cv2.imwrite(str(out / "setup_closeup.png"), cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
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
                    help="override a constant for an experiment (a Python literal): --set PUSH_HEIGHT=0.001, or a part's with --set tp.PAD_SPRING=4000.0")
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
                                                               "final_stage", "rail_shift_mm", "rail_turn_deg",
                                                               "wall_s")}))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
