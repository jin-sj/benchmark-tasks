"""Scripted demo for the fix-the-unlatched-block task, closed loop on the simulator's object poses.

    python demo.py                    # right arm presses the faulty block's near end down until it latches: 1.0
    python demo.py --variant hold     # no arm moves: the faulty block must stay as it started (0.0)
    python demo.py --device cpu       # CPU physics, ~3x faster, for quick checks

It drives the ``yam_arena`` embodiment directly (20 Hz control, object_jitter 0, as a trial runs) and reads the
task's own scoring from each step's ``info``. The script knows which block is faulty (a policy has to see it):

- right arm: fingers closed, comes down over the faulty block's near end (toward the arms), 20 mm from its middle,
  and presses down, the command leading (the arm is soft), until the block latches; holds 0.5 s; lifts away.
- left arm: stays where it is.

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
    "One of the terminal blocks on the DIN rail is not latched. Find it and press it home so it latches, without "
    "unlatching or moving the other blocks."
)
VARIANTS = ("full", "hold")
CAMERAS = ("top_cam", "left_cam", "right_cam")
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ
MAX_STEPS = 600
MM = 1e-3

PAD_TIP = 0.013  # pad tips beyond the TCP
PRESS_X = -20 * MM  # press here on the block (block frame x: toward the arms from its middle; the near end)
PRESS_RATE, PRESS_MAX = 20 * MM, 40 * MM  # the press command leads by this much more per s, at most this far
HOLD_S = 0.5  # keep pressing this long once latched
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
    obs = emb.reset(Scene(id=f"fix-unlatched-{variant}", instruction=INSTRUCTION), seed=0)
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

    # -- right arm: press the faulty block's near end down ------------------------------------------------
    def right_script():
        arm = arms["right"]
        if variant == "hold":
            phase["right"] = "idle"
            yield from idle()
        arm.grip = 0.0
        down = rot_z(rail_yaw()) @ arm.down
        o, r = block()
        top = o + r @ np.array([PRESS_X, 0.0, tp.BLOCK_HEIGHT])  # roughly the top of its near end
        tcp = top + np.array([0, 0, PAD_TIP])
        yield from move(arm, tcp + (0, 0, 0.06), 2.5, grip=0.0, rot=down, label="over the near end")
        yield from move(arm, tcp + (0, 0, 0.01), 1.2, label="down near it")
        yield from settle(arm, tcp + (0, 0, 0.01), 0.6, "line up over it")
        start = arm.pos.copy()
        phase["right"] = "press down"
        lead, held = 0.0, 0
        for _ in range(round(8.0 / DT)):
            if task.stage(world["states"]) == "latched":
                report.setdefault("latched_step", world["step"])
                held += 1
                if held >= round(HOLD_S / DT):
                    break
            lead = min(lead + PRESS_RATE * DT, PRESS_MAX + 0.01)
            arm.pos = start - np.array([0, 0, lead])
            yield
        report["press_lead_mm"] = round(lead / MM, 1)
        yield from move(arm, start + (0, 0, 0.05), 1.2, label="lift away")
        phase["right"] = "done"
        report["script_done_step"] = world["step"]
        yield from idle()

    scripts = {"left": idle(), "right": right_script()}
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
                      "spread_mm": [round(t / MM, 2) for t in b["feet_spread"]], "tilt": round(b["tilt_deg"], 2),
                      "pads_mm": [round(t / MM, 2) for t in b["pads_squeeze"]], "latched": task.latched(b)}
        for side in ("left", "right"):
            row[f"{side[0]}_cmd"] = [round(float(v), 4) for v in arms[side].pos]
            row[f"{side[0]}_meas"] = [round(float(v), 4) for v in measured_tcp(side)[0]]
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
        "start_tilt_deg": log[0][task.TARGET]["tilt"],
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
                                                               "final_stage", "undisturbed", "start_tilt_deg",
                                                               "wall_s")}))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
