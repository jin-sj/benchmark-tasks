"""Scripted demo for the price-labeler task. It uses only the task's known starting layout and the robot's own joint
readings (the measured gripper positions), never the simulator's object poses.

    python demo.py                    # pick up the labeler, price all six cans: 1.0
    python demo.py --variant three    # price three cans only: 0.5
    python demo.py --variant double   # the first can gets two labels (and one can none): 0.5
    python demo.py --variant spoil    # one label spoiled by a second squeeze, all six cans priced: 0.5
    python demo.py --variant hold     # no arm moves: 0
    python demo.py --device cpu       # CPU physics, ~3x faster, for quick checks

The right arm, its gripper pointing down with its fingers closing across the labeler's handle and trigger, takes the
labeler out of its stand. For each can it squeezes the trigger fully and lets it go (a label comes out under the
nose roller), brings the roller over the middle of the can's lid, presses it on and lifts it off.

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
INSTRUCTION = "Price the cans: stick one price label flat on each can's lid, inside the lid's seam ring."
VARIANTS = ("full", "hold", "three", "double", "spoil")
CAMERAS = ("top_cam", "left_cam", "right_cam")
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ
MAX_STEPS = 1200
MM = 1e-3

OPEN = 0.55  # gripper opening (0..1 of 94 mm): ~52 mm, round the handle and the trigger (32 mm)
HOLD = 0.30  # holding the labeler: the fingers pressing the trigger ~3 mm in (about 12 N), short of feeding a label
FIRE = 0.0  # squeezing the trigger fully in
HOVER = 15 * MM  # bring the roller this far above the lid, then press it on
PRESS = 3 * MM  # press the roller this far into the lid (the arm gives)
AFTER_SUCCESS_S = 4.0

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
    obs = emb.reset(Scene(id=f"price-labeler-{variant}", instruction=INSTRUCTION), seed=0)
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

    lp = task.lp
    arm = arms["right"]
    hold_rot = rot_z(math.pi / 2) @ arm.down  # pointing down, the fingers closing across the handle and the trigger
    grip_at = np.array([(lp.HANDLE[0] + lp.TRIGGER[1]) / 2, 0.0, 0.0])  # the pinch's middle, in the labeler's frame
    lab0 = np.asarray(task.LABELER_ORIGIN, dtype=float)
    nose_off = lp.NOSE - grip_at  # from the TCP to the nose roller (the labeler held level, its handle along y)

    def tcp_for_nose(nose):
        return np.asarray(nose, dtype=float) - nose_off

    def squeeze():
        yield from move(arm, arm.pos, 0.5, grip=FIRE, label="squeeze the trigger")
        yield from move(arm, arm.pos, 0.5, grip=HOLD, label="let the trigger go")

    def script():
        phase["left"] = "idle"
        if variant == "hold":
            phase["right"] = "idle"
            yield from idle()
        # pick the labeler up out of its stand
        g = lab0 + grip_at
        yield from move(arm, g + (0, 0, 0.06), 2.5, grip=OPEN, rot=hold_rot, label="over the labeler")
        yield from settle_until(arm, g + (0, 0, 0.06), "line up over the labeler")
        corr = arm.pos - (g + (0, 0, 0.06))
        yield from move(arm, g + corr, 1.2, label="down to its handle")
        yield from settle_until(arm, g, "line up on the handle")
        yield from move(arm, arm.pos, 0.8, grip=HOLD, label="take hold of it")
        yield from move(arm, arm.pos + (0, 0, 0.09), 1.5, label="lift it out")
        report["picked_step"] = world["step"]
        cans = task.CAN_STARTS if variant != "three" else task.CAN_STARTS[:3]
        for k, (cx, cy) in enumerate(cans):
            if variant == "double" and k == 1:  # a second label on the first can
                cx, cy = cans[0]
            if variant == "spoil" and k == 0:  # an extra squeeze: the first label is spoiled
                yield from squeeze()
            yield from squeeze()
            lid = np.array([cx, cy, task.LID_TOP])
            above = tcp_for_nose(lid + (0, 0, HOVER))
            yield from move(arm, above, 1.5, label=f"over can {k + 1}")
            yield from settle_until(arm, above, f"line up over can {k + 1}", tol=1.0 * MM, max_s=1.5)
            corr = arm.pos - above
            yield from move(arm, tcp_for_nose(lid - (0, 0, PRESS)) + corr, 0.8, label=f"press it on can {k + 1}")
            yield from move(arm, arm.pos, 0.3, label=f"press it on can {k + 1}")
            yield from move(arm, above + corr, 0.6, label=f"off can {k + 1}")
            report.setdefault("pressed", []).append(world["step"])
        yield from move(arm, arm.pos + (0.0, 0.0, 0.04), 0.6, label="away from the cans")
        yield from move(arm, g + (0, 0, 0.10), 2.0, label="back over the stand")
        phase["right"] = "done"
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
        per_can, stray = task.labels_on_cans(st)
        row = {"step": step, "left": phase["left"], "right": phase["right"], "score": res.info.get("score"),
               "success": res.info.get("success"), "trigger_mm": round(st["labeler"].trigger_mm, 2),
               "labeler": [round(float(v), 4) for v in st["labeler"].origin], "fed": task.RECORD.get("fed"),
               "out": task.RECORD.get("out"), "per_can": per_can, "stray": stray}
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
    per_can, stray = task.labels_on_cans(st)
    summary = {
        "variant": variant,
        "task_score": 1.0 if success else float(result.info.get("score", 0.0)),
        "terminated": bool(result.terminated),
        "termination_reason": result.termination_reason,
        "steps": len(log),
        "run_limit_steps": MAX_STEPS,
        "priced": task.priced(st),
        "per_can": per_can,
        "stray": stray,
        "stuck": task.RECORD.get("stuck"),
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
                                                               "priced", "stray", "wall_s")}))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
