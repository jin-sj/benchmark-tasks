"""Scripted two-arm demo for the test-probes task. It uses only the task's known starting layout and the robot's own
joint readings (the measured gripper position), never the simulator's object poses.

    python demo.py                    # each arm takes a probe and holds its tip in the L or N test opening: 1.0
    python demo.py --variant hold     # no arm moves (0.0)
    python demo.py --variant wrong    # the right arm's probe goes into a terminal two blocks from N first, then N: 0
    python demo.py --device cpu       # CPU physics, ~3x faster, for quick checks

Each arm, the gripper pointing down with its fingers closing across the rail (so the two grippers' narrow sides face
each other over the strip): comes down over its probe's handle, closes on it 80 mm above the tip (near its centre of mass), lifts it out of its
holder, carries it over its terminal's test opening, lines its measured gripper position up over the opening and
lowers the tip until it is 5 mm into it, feeling for the opening: if the gripper stops descending early (the tip is
on the block's top), it lifts the tip 4 mm, shifts it 1 or 2 mm and tries again. Then it holds there. The left arm takes the red probe to L, the right arm the black probe to N.

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
    "Measure the voltage between the L terminal and the N terminal: hold one probe's tip in each terminal's test "
    "opening, both at once, for one second. Do not put a probe tip into any other terminal's opening."
)
VARIANTS = ("full", "hold", "wrong")
CAMERAS = ("top_cam", "left_cam", "right_cam")
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ
MAX_STEPS = 800
MM = 1e-3

PAD_TIP = 0.013  # pad tips beyond the TCP
OPEN = 0.35  # gripper opening (0..1 of 94 mm): ~33 mm between the pads, around the 10 mm handle
GRIP_AT = 80 * MM  # grip the handle this far above the probe's tip (the TCP): near the probe's centre of mass, so
# it does not pivot in the pinch (gripped low, the top-heavy probe tips over in the grip)
TIP_ABOVE = 10 * MM  # line the tip up this far above the opening's mouth
INSERT = 5 * MM  # then lower the tip this far below the mouth (it is in the bore) ...
PRESS = 1.5 * MM  # ... and on down onto the clamp (the opening's floor), holding it there pressed this much
LOWER_RATE = 6 * MM  # per second
MAX_LEAD = 3 * MM  # while lowering, the command leads the measured gripper by at most this (no hard push on the top)
STALL_STEPS = 6  # the gripper has stopped descending for this many steps: the tip is on the block's top
SEARCH = [(0, 0)] + [(r * MM * c, r * MM * s_) for r in (1.0, 2.0) for c, s_ in ((1, 0), (0, 1), (-1, 0), (0, -1))]
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
    obs = emb.reset(Scene(id=f"test-probes-{variant}", instruction=INSTRUCTION), seed=0)
    set_rendering(emb, video)
    task = emb.task.module
    tp, pp = task.tp, task.pp
    so = np.asarray(task.STRIP_ORIGIN, dtype=float)
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

    def probe_script(side, probe, targets):
        """Take ``probe`` from its holder and hold its tip in each opening of ``targets`` in turn (the last one until
        the end)."""
        arm = arms[side]
        if variant == "hold":
            phase[side] = "idle"
            yield from idle()
        # fingers closing across the rail, each arm turned the way that keeps its last wrist joint in range
        down = rot_z((-1 if side == "left" else 1) * math.pi / 2) @ arm.down
        tip0 = np.asarray(pp.probe_in_holder(task.HOLDERS[task.PROBES[probe]]), dtype=float)
        g = tip0 + np.array([0.0, 0.0, GRIP_AT])
        yield from move(arm, g + (0, 0, 0.08), 2.5, grip=OPEN, rot=down, label="over the probe")
        yield from settle_until(arm, g + (0, 0, 0.08), "line up over the probe")
        corr = arm.pos - (g + (0, 0, 0.08))
        yield from move(arm, g + corr, 1.5, label="down around the handle")
        yield from settle_until(arm, g, "line up on the handle")
        yield from move(arm, arm.pos, 0.8, grip=0.0, label="grip the handle")
        yield from move(arm, arm.pos + (0, 0, 0.05), 1.2, label="lift it out")
        report.setdefault("lifted_step", {})[side] = world["step"]
        for k, i in enumerate(targets):
            o = so + np.array(pp.opening(i))
            above = o + np.array([0.0, 0.0, TIP_ABOVE + GRIP_AT])
            yield from move(arm, above + (0, 0, 0.03), 2.0, label=f"over terminal {i + 1}")
            yield from settle_until(arm, above, f"line up over terminal {i + 1}", tol=0.4 * MM, hold=6, max_s=4.0)
            # feel for the opening: lower the tip; if the gripper stops descending before the tip is in (the tip is
            # on the block's top), lift it, shift it a little and try again (all from the arm's own joint readings)
            goal_z = o[2] - INSERT + GRIP_AT
            entered = False
            for n_try, (dx, dy) in enumerate(SEARCH):
                phase[side] = f"into terminal {i + 1}" + (f" (try {n_try + 1})" if n_try else "")
                start = above + np.array([dx, dy, 0.0])
                yield from settle_until(arm, start, phase[side], tol=0.5 * MM, hold=3, max_s=2.0)
                cmd_z, stall, last_z = arm.pos[2], 0, measured_tcp(side)[0][2]
                for _ in range(round(4.0 / DT)):
                    p = measured_tcp(side)[0]
                    if p[2] <= goal_z + 0.3 * MM:
                        entered = True
                        break
                    stall = stall + 1 if last_z - p[2] < 0.2 * MM else 0
                    last_z = p[2]
                    if stall >= STALL_STEPS and cmd_z < p[2] - 1 * MM:  # stopped above the goal: on the top
                        break
                    cmd_z = max(cmd_z - LOWER_RATE * DT, p[2] - MAX_LEAD)
                    arm.pos = np.array([start[0] + 0.1 * (start[0] - p[0]), start[1] + 0.1 * (start[1] - p[1]), cmd_z])
                    yield
                if entered:
                    break
                yield from move(arm, measured_tcp(side)[0] + (0, 0, 0.004), 0.4, label="lift the tip off")
            # on down until the tip stops on the clamp, then hold it there, pressed lightly
            phase[side] = f"onto the clamp of terminal {i + 1}"
            cmd_z, stall, last_z = arm.pos[2], 0, measured_tcp(side)[0][2]
            for _ in range(round(3.0 / DT)):
                p = measured_tcp(side)[0]
                stall = stall + 1 if last_z - p[2] < 0.1 * MM else 0
                last_z = p[2]
                if stall >= STALL_STEPS:
                    break
                cmd_z = max(cmd_z - LOWER_RATE * DT, p[2] - MAX_LEAD)
                arm.pos = np.array([arm.pos[0], arm.pos[1], cmd_z])
                yield
            arm.pos = np.array([arm.pos[0], arm.pos[1], measured_tcp(side)[0][2] - PRESS])
            report.setdefault("tries", {})[f"{side}_{i + 1}"] = n_try + 1
            report.setdefault("inserted", []).append({"side": side, "terminal": i + 1, "step": world["step"]})
            if k < len(targets) - 1:  # out again, up to the next one
                yield from move(arm, arm.pos + (0, 0, 0.025), 1.0, label=f"out of terminal {i + 1}")
        phase[side] = f"hold it in terminal {targets[-1] + 1}"
        yield from idle()


    wrong = [pp.N_INDEX - 2] if variant == "wrong" else []  # two blocks from N (10.4 mm): a missed tip cannot slide into N
    scripts = {"left": probe_script("left", pp.PROBE_RED, [pp.L_INDEX]),
               "right": probe_script("right", pp.PROBE_BLACK, wrong + [pp.N_INDEX])}
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
               "success": res.info.get("success"), "wrong": pp.RECORD.get("wrong"), "both_s": round(pp.RECORD.get("both_s", 0.0), 2),
               "probes": pp.RECORD.get("now")}
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
        "wrong": pp.RECORD.get("wrong"),
        "seated_ever": pp.RECORD.get("seated_ever"),
        "lift_mm": {k: round(v / MM, 1) for k, v in pp.RECORD.get("lift", {}).items()},
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
                                                               "wrong",
                                                               "wall_s")}))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
