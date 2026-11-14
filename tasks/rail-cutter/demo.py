"""Scripted two-arm demo for the rail-cutter task. It uses only the task's known starting layout and the robot's own
joint readings (the measured gripper positions), never the simulator's object poses.

    python demo.py                    # set the rail at 250 mm, shut the clamp, press the cut button: 1.0
    python demo.py --variant short    # the rail is set 3 mm short (247 mm) before the cut: 0.5
    python demo.py --variant noclamp  # the button is pressed with the clamp open (the cutter does not cut): 0.25
    python demo.py --variant nocut    # the rail is set at 250 mm and clamped, the button never pressed: 0.25
    python demo.py --variant hold     # no arm moves: 0
    python demo.py --device cpu       # CPU physics, ~3x faster, for quick checks

- left arm: the gripper pointing down, its fingers closing across the rail, grips the rail by its lips 100 mm from
  its left end, slides it 290 mm toward the outfeed (lining its measured gripper position up to within 0.3 mm), which
  puts the rail's right end 250 mm past the blade, and holds it there;
- right arm: once the rail is set, grips the clamp's handle across its sides, pushes it down along its arc past its
  over-centre point until it is shut on the rail, lets go, then presses the cut button with its closed fingertips.

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
INSTRUCTION = "Cut a 250 mm length off the right end of the rail with the rail cutter."
VARIANTS = ("full", "hold", "short", "noclamp", "nocut")
CAMERAS = ("top_cam", "left_cam", "right_cam")
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ
MAX_STEPS = 800
MM = 1e-3

PAD_TIP = 0.013  # pad tips beyond the TCP
OPEN = 0.5  # gripper opening (0..1 of 94 mm): ~47 mm between the pads, around the 35 mm rail
RAIL_GRIP_FROM_END = 100 * MM  # grip the rail this far from its left end
PAD_BELOW_LIPS = 1.8 * MM  # the pad tips this far below the lips' tops (on the lips, clear of the guide fences)
SHORT_BY = 3 * MM  # "short": set the rail this much short of the mark
HANDLE_OPEN = 0.35  # ~33 mm between the pads, around the 22 mm handle
PULL_RATE_DEG = 15.0  # sweep the handle's target down its arc this fast
LEAD_DOWN = 15 * MM  # the command leads the handle's arc point downward by up to this (the clamp's spring pushes back)
CLAMP_DOWN_DEG = 3.0  # push the clamp's handle down until the gripper says the clamp is this far from level
BUTTON_PUSH = 16 * MM  # press the button: the command this far below its cap's top (it stops at 5 mm in; the arm gives)
AFTER_SUCCESS_S = 3.0


def _rail_pose(task, st) -> list:
    """Diagnostics: the rail origin's x and z offsets from the cutter's die (mm), its yaw and pitch (degrees)."""
    from yam_arena.urdf import quat_xyzw_to_matrix

    r = st["rail"]
    rot = quat_xyzw_to_matrix(np.asarray(r.quat_xyzw, dtype=float))
    o = np.asarray(r.pos, dtype=float) - rot @ task.C_RAIL
    c = np.asarray(task.CUTTER_ORIGIN, dtype=float)
    return [round((o[0] - c[0]) / MM, 2), round((o[2] - c[2] - task.ct.LEDGE_TOP) / MM, 2),
            round(float(np.degrees(np.arctan2(-rot[0, 1], rot[1, 1]))), 2),
            round(float(np.degrees(np.arctan2(rot[2, 1], rot[1, 1]))), 2)]


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
    obs = emb.reset(Scene(id=f"rail-cutter-{variant}", instruction=INSTRUCTION), seed=0)
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

    def settle_until(arm, target, label, tol=0.6 * MM, hold=4, max_s=3.0, gain=0.15, limit=0.015):
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

    ct = task.ct
    c0 = np.asarray(task.CUTTER_ORIGIN, dtype=float)

    def handle_world(deg):
        x, z = ct.handle_point(deg)
        return c0 + np.array([x, ct.CLAMP_Y, z])

    def lever_deg_from(p):
        """The clamp's angle the gripper's position implies (the handle's middle is between the pads)."""
        return math.degrees(math.atan2(p[2] - (c0[2] + ct.PIVOT[1]), (c0[0] + ct.PIVOT[0]) - p[0]))

    def rail_script():
        arm = arms["left"]
        if variant == "hold":
            phase["left"] = "idle"
            yield from idle()
        down = rot_z(-math.pi / 2) @ arm.down  # fingers closing across the rail (the left wrist's range allows -90)
        r0 = np.asarray(task.RAIL_ORIGIN, dtype=float)
        left_end_y = r0[1] + tp.RAIL_LENGTH / 2
        g = np.array([r0[0], left_end_y - RAIL_GRIP_FROM_END, r0[2] + tp.RAIL_HEIGHT - PAD_BELOW_LIPS + PAD_TIP])
        yield from move(arm, g + (0, 0, 0.06), 2.5, grip=OPEN, rot=down, label="over the rail")
        yield from settle_until(arm, g + (0, 0, 0.06), "line up over the rail")
        corr = arm.pos - (g + (0, 0, 0.06))
        yield from move(arm, g + corr, 1.5, label="down around the rail")
        yield from settle_until(arm, g, "line up on the rail")
        yield from move(arm, arm.pos, 0.8, grip=0.0, label="grip the rail")
        travel = task.START_GAP + task.TARGET - (SHORT_BY if variant == "short" else 0.0)
        goal = g - np.array([0.0, travel, 0.0])
        yield from move(arm, arm.pos - np.array([0.0, travel, 0.0]), 4.0, label="slide it through the die")
        yield from settle_until(arm, goal, "set it at the mark", tol=0.3 * MM, hold=8, max_s=4.0, gain=0.1)
        report["rail_set_step"] = world["step"]
        flags.add("rail_set")
        phase["left"] = "hold the rail"
        yield from idle()

    def lever_script():
        arm = arms["right"]
        if variant == "hold":
            phase["right"] = "idle"
            yield from idle()
        yield from wait_for(arm, "rail_set", "wait for the rail")
        if variant != "noclamp":
            # shut the clamp: grip its handle, push it down past its over-centre point, let go (it stays shut)
            h = handle_world(ct.UP_DEG) + np.array([0.0, 0.0, PAD_TIP])
            yield from move(arm, h + (0, 0, 0.05), 2.0, grip=HANDLE_OPEN, rot=arm.down, label="over the clamp's handle")
            yield from settle_until(arm, h + (0, 0, 0.05), "line up over the handle")
            corr = arm.pos - (h + (0, 0, 0.05))
            yield from move(arm, h + corr, 1.2, label="down to the handle")
            yield from settle_until(arm, h, "line up on the handle")
            yield from move(arm, arm.pos, 0.8, grip=0.0, label="grip the handle")
            phase["right"] = "shut the clamp"
            a = ct.UP_DEG
            for _ in range(round(8.0 / DT)):
                p = measured_tcp("right")[0]
                if lever_deg_from(p - np.array([0.0, 0.0, PAD_TIP])) <= CLAMP_DOWN_DEG:
                    break
                a = max(a - PULL_RATE_DEG * DT, -5.0)
                lead = min(LEAD_DOWN, max(0.0, (ct.UP_DEG - a) / ct.UP_DEG * LEAD_DOWN))
                arm.pos = handle_world(a) + np.array([0.0, 0.0, PAD_TIP - lead])
                yield
            report["clamp_shut_step"] = world["step"]
            yield from move(arm, arm.pos, 0.6, grip=HANDLE_OPEN, label="let go of the clamp")
            yield from move(arm, arm.pos + (0, 0, 0.06), 1.0, label="away from the clamp")
        if variant == "nocut":
            phase["right"] = "done"
            report["script_done_step"] = world["step"]
            yield from idle()
        # press the cut button: the gripper closed, pointing down, its fingertips on the button's cap
        top = c0 + ct.button_top() + np.array([0.0, 0.0, PAD_TIP])
        yield from move(arm, top + (0, 0, 0.04), 2.0, grip=0.0, rot=arm.down, label="over the cut button")
        yield from settle_until(arm, top + (0, 0, 0.04), "line up over the button")
        corr = arm.pos - (top + (0, 0, 0.04))
        yield from move(arm, top + corr + (0, 0, -BUTTON_PUSH), 1.0, label="press the cut button")
        yield from move(arm, arm.pos, 0.5, label="press the cut button")
        report["button_step"] = world["step"]
        yield from move(arm, arm.pos + (0, 0, 0.05), 0.8, label="off the button")
        yield from move(arm, arm.pos, 2.0, label="wait for the stroke")
        phase["right"] = "done"
        report["script_done_step"] = world["step"]
        yield from idle()

    scripts = {"left": rail_script(), "right": lever_script()}
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
               "success": res.info.get("success"), "clamp_deg": round(st["cutter"].clamp_deg, 2),
               "button_mm": round(st["cutter"].button_mm, 2), "blade_mm": round(st["cutter"].blade_mm, 1),
               "strokes": st["cutter"].strokes,
               "rail_past_mm": round(task._rail_end(st)[0] / MM, 2), "in_die": task._rail_end(st)[1], "rail": _rail_pose(task, st),
               "cut": task.RECORD.get("cut")}
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
    summary = {
        "variant": variant,
        "task_score": 1.0 if success else float(result.info.get("score", 0.0)),
        "terminated": bool(result.terminated),
        "termination_reason": result.termination_reason,
        "steps": len(log),
        "run_limit_steps": MAX_STEPS,
        "cut": task.RECORD.get("cut"),
        "best_err_mm": None if task.RECORD.get("best") is None else round(task.RECORD["best"] / MM, 2),
        "set_at_mark": bool(task.RECORD.get("set_ok")),
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
                                                               "cut",
                                                               "wall_s")}))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
