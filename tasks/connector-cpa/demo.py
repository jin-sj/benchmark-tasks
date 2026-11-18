"""Scripted demo for the connector-cpa task. It uses only the task's known starting layout and the robot's own joint
readings (the measured gripper positions), never the simulator's object poses.

    python demo.py                    # plug on, latched, CPA home: 1.0
    python demo.py --variant nocpa    # plug on and latched, CPA left in pre-lock: 0.5
    python demo.py --variant half     # plug pushed only part way (onto the lock ramp) and left: half-mated, 0
    python demo.py --variant halfcpa  # plug left half-mated, then the CPA pushed as hard as for home: 0
    python demo.py --variant hold     # no arm moves: 0
    python demo.py --device cpu       # CPU physics, ~3x faster, for quick checks

The right arm, pointing down, its fingers closing across the plug's collar, picks the plug up off the table, carries
it over the header and lowers it into the shroud's mouth until it rests on the seal. It lets go, closes its fingers
and pushes the plug's top straight down with its fingertips, 3 mm toward the latch side of its middle, until the
plug is fully on (the latch clicks over the lock ramp; about 33 N of steady push), then pushes the CPA home.
The push runs the arm's command ahead of its measured position (the arm gives under load) and stops when the measured
fingertips are down at the seated height.

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
INSTRUCTION = "Connect the harness plug to the sensor and close its orange CPA lock."
VARIANTS = ("full", "nocpa", "half", "halfcpa", "hold")
CAMERAS = ("top_cam", "left_cam", "right_cam")
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ
MAX_STEPS = 1800
MM = 1e-3

OPEN = 0.40  # gripper opening (0..1 of 94 mm): ~38 mm, round the 24.8 mm collar
NEAR = 0.29  # ~27 mm: closed in to just clear of the collar first, so the squeeze cannot shove the plug aside
SHUT = 0.0  # fingers closed, for pushing with the fingertips
GRIP = 0.10  # ~9.4 mm: holding the 24.8 mm collar with each finger squeezing about 38 N (5,000 N/m drive x 7.7 mm
# short of its target; at most 40 N, the YAM's finger force), so the two fingers centre the plug between them
GRIP_Z = 27 * MM  # the TCP this far above the plug's face when gripping its collar
TIP = 13 * MM  # the fingertips this far below the TCP
PUSH_X = -3 * MM  # push the plug's top this far off its middle (task x): toward the latch side
SWING_X = 2.4 * MM  # the held plug hangs tilted a few degrees by its lead, its face about this far back (-x) of the TCP
PUSH_SPEED = 20 * MM  # m/s, commanded
PUSH_OVER = 150 * MM  # the most the command may run ahead of the measured fingertips (the arm gives under load)
AFTER_SUCCESS_S = 3.0
LOOK_POS = (0.25, -0.05, 0.16)  # the left wrist camera while it watches (x, y, height above the table)
LOOK_AT = (0.0, 0.0, 0.085)  # ... pointing at this point (offset from the header's axis; height above the table)


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


def make_embodiment(video: bool, device: str = "cuda:0"):
    from yam_arena.inspect_embodiment import YamArenaEmbodiment

    size = (360, 640) if video else (64, 64)
    return YamArenaEmbodiment(task=str(TASK), control_hz=CONTROL_HZ, object_jitter=0.0, cam_height=size[0],
                              cam_width=size[1], device=device)


def set_rendering(emb, on: bool) -> None:
    emb._env.unwrapped.cfg.sim.render_interval = emb._env.unwrapped.cfg.decimation if on else 10**9


def run(variant: str, out: Path, video: bool, max_steps: int, emb=None) -> dict:
    from inspect_robots import Scene
    from inspect_robots.types import Action
    from inspect_robots_yam.packing import pack, split

    from yam_arena import ik, station
    from yam_arena.demo import grip_rotation

    emb = emb or make_embodiment(video)
    obs = emb.reset(Scene(id=f"connector-cpa-{variant}", instruction=INSTRUCTION), seed=0)
    set_rendering(emb, video)
    task = emb.task.module
    cp = task.cp
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

    def settle_until(arm, target, label, tol=0.8 * MM, hold=4, max_s=3.0, gain=0.15, limit=0.02):
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

    def press(arm, xy, z_goal, label, max_s=10.0, hold_s=0.6, max_force=None):
        """Push straight down at ``xy`` until the measured TCP is down to ``z_goal``: the command runs down at
        PUSH_SPEED, at most PUSH_OVER ahead of the measured TCP, keeping x and y on ``xy`` by the measured error.
        Then hold there for ``hold_s`` and ease off. Returns via report the lowest measured TCP height."""
        phase[arm.side] = label
        xy = np.asarray(xy, dtype=float)
        corr = np.zeros(2)
        low, held = math.inf, 0
        for _ in range(round(max_s / DT)):
            m = measured_tcp(arm.side)[0]
            low = min(low, m[2])
            if m[2] <= z_goal:  # there: hold the push where it is
                held += 1
                if held * DT >= hold_s:
                    break
                yield
                continue
            corr = np.clip(corr + 0.15 * (xy - m[:2]), -0.03, 0.03)
            z = max(arm.pos[2] - PUSH_SPEED * DT, m[2] - PUSH_OVER)
            if max_force is not None:  # a force-limited push: no further down while pushing this hard
                f_down = -float(tcp_force(arm.side)[2])
                report["max_push_n"] = max(report.get("max_push_n", 0.0), f_down)
                if f_down >= max_force:
                    z = arm.pos[2]
                    held += 1
                    if held * DT >= 2.0:
                        break
            arm.pos = np.array([xy[0] + corr[0], xy[1] + corr[1], z])
            yield
        report.setdefault("press", []).append({"label": label, "low": low, "goal": z_goal,
                                               "step": world["step"]})
        m = measured_tcp(arm.side)[0]
        yield from move(arm, np.array([arm.pos[0], arm.pos[1], m[2] + 4 * MM]), 0.8, label=label + ": ease off")

    def idle():
        while True:
            yield

    from yam_arena.constants import SHOULDER, WRIST

    gains = np.array([SHOULDER["stiffness"]] * 3 + [WRIST["stiffness"]] * 3)
    caps = np.array([SHOULDER["effort"]] * 3 + [WRIST["effort"]] * 3)

    def tcp_force(side: str) -> np.ndarray:
        """The force the arm's TCP pushes with (N, world), from its joint drives' stiffness and position errors
        (static; gravity is fed forward separately), by the Jacobian of its kinematics."""
        left, right = split(np.asarray(world["obs"].state["joint_pos"]))
        q = (left if side == "left" else right)[:6]
        tau = np.clip(gains * (arms[side].q - q), -caps, caps)
        p0, r0 = ik.tcp_pose(q, arms[side].base)
        jac = np.zeros((6, 6))
        for i in range(6):
            dq = q.copy()
            dq[i] += 1e-5
            p1, r1 = ik.tcp_pose(dq, arms[side].base)
            jac[:3, i] = (p1 - p0) / 1e-5
            jac[3:, i] = ik._rot_error(r1, r0) / 1e-5
        wrench = np.linalg.lstsq(jac.T, tau, rcond=None)[0]
        return wrench[:3]

    arm = arms["right"]
    down = arm.down  # fingers closing along world y: across the plug's collar
    plug0 = np.asarray(task.PLUG_AT, dtype=float)
    floor = np.asarray(task.HEADER_AT, dtype=float) + (0.0, 0.0, cp.FLOOR * MM)  # the cavity floor's middle
    hxy = floor[:2]
    seat_top = floor[2] + (cp.SEAT + cp.COLLAR[3]) * MM  # the plug's top, seated
    # the CPA's middle, seen from above: the connector's +x is the task's -x
    cpa_xy = hxy + np.array([-(cp.CPA[0] + cp.CPA[1]) / 2 * MM, 0.0])
    cpa_home_top = floor[2] + (cp.SEAT + cp.CPA[4] - cp.CPA_TRAVEL) * MM

    # the left arm: its wrist camera watching the header from the front left, out of the top camera's way
    # the left arm: parked front left, turned so that its wrist camera looks at the header, image up roughly up
    cam = station.wrist_camera("left")
    from yam_arena.urdf import quat_xyzw_to_matrix

    cam_rot = quat_xyzw_to_matrix(np.asarray(cam.rot_xyzw))
    f_g, u_g = -cam_rot[:, 2], cam_rot[:, 1]  # the camera's view and up, in the gripper frame (OpenGL camera)
    cam_c = np.array([LOOK_POS[0], LOOK_POS[1], task.TOP + LOOK_POS[2]])
    target = np.array([task.HEADER_AT[0] + LOOK_AT[0], task.HEADER_AT[1] + LOOK_AT[1], task.TOP + LOOK_AT[2]])
    f_w = (target - cam_c) / np.linalg.norm(target - cam_c)
    u_w = np.array([0.0, 0.0, 1.0]) - f_w[2] * f_w
    u_w /= np.linalg.norm(u_w)
    look_rot = np.column_stack([f_w, u_w, np.cross(f_w, u_w)]) @ np.column_stack([f_g, u_g, np.cross(f_g, u_g)]).T
    from yam_arena.constants import TCP_OFFSET

    look = cam_c + look_rot @ (np.asarray(TCP_OFFSET) - np.asarray(cam.pos))

    def watch():
        yield from move(arms["left"], look, 3.0, grip=0.0, rot=look_rot, label="watch the header")
        phase["left"] = "watching"
        yield from idle()

    def script():
        if variant == "hold":
            phase["right"] = "idle"
            yield from idle()
        # take the plug by its collar
        g = plug0 + (0.0, 0.0, GRIP_Z)
        yield from move(arm, g + (0, 0, 0.06), 2.5, grip=OPEN, rot=down, label="over the plug")
        yield from settle_until(arm, g + (0, 0, 0.06), "line up over the plug")
        corr = arm.pos - (g + (0, 0, 0.06))
        yield from move(arm, g + corr, 1.2, label="down to its collar")
        yield from settle_until(arm, g, "line up on the collar")
        yield from move(arm, arm.pos, 0.6, grip=NEAR, label="close in on the collar")
        yield from move(arm, arm.pos, 0.6, grip=GRIP, label="grip the collar")
        report["gripped_step"] = world["step"]
        yield from move(arm, arm.pos + (0, 0, 0.10), 1.5, label="lift it")
        # over the header, the plug's face 4 mm above the rim, then into the mouth
        above = np.array([hxy[0] + SWING_X, hxy[1], floor[2] + (cp.SHROUD_H + 4.0) * MM + GRIP_Z])
        yield from move(arm, above + (0, 0, 0.02), 2.0, label="over the header")
        yield from settle_until(arm, above, "line up over the header", tol=0.6 * MM, max_s=4.0)
        corr = arm.pos - above
        rest = np.array([hxy[0] + SWING_X, hxy[1], floor[2] + (cp.SEAL_TOP + 0.5) * MM + GRIP_Z])
        yield from move(arm, rest + corr, 1.5, label="into the shroud")
        yield from settle_until(arm, rest, "onto the seal", tol=1.0 * MM, max_s=1.5)
        yield from move(arm, arm.pos, 0.6, grip=0.65, label="let go")
        yield from move(arm, arm.pos, 0.4, label="let go")
        report["inserted_step"] = world["step"]
        if STOP_AFTER_INSERT:  # mechanism tests: leave the plug resting in the shroud's mouth
            yield from move(arm, arm.pos + (0, 0, 0.06), 1.0, label="away (test)")
            phase["right"] = "done"
            yield from idle()
        m = measured_tcp("right")[0]
        top = m[2] - GRIP_Z + cp.COLLAR[3] * MM  # the plug's top now
        push_xy = hxy + (PUSH_X, 0.0)
        yield from move(arm, np.array([push_xy[0], push_xy[1], top + TIP + 0.02]), 1.0, label="up, fingers shut")
        yield from move(arm, arm.pos, 0.6, grip=SHUT, label="shut the fingers")
        yield from settle_until(arm, np.array([push_xy[0], push_xy[1], top + TIP + 0.006]), "over the plug's top",
                                tol=0.8 * MM, max_s=2.0)
        goal = seat_top + TIP + 0.6 * MM
        if variant in ("half", "halfcpa"):  # only onto the lock ramp: the face 3 mm above its seat
            goal = seat_top + TIP + 2.0 * MM
        yield from press(arm, push_xy, goal, "push the plug on")
        report["pushed_step"] = world["step"]
        yield from move(arm, arm.pos + (0, 0, 0.02), 0.8, label="off the plug")
        if variant in ("full", "halfcpa"):  # halfcpa: the CPA pushed on a half-mated plug
            over = np.array([cpa_xy[0], cpa_xy[1], cpa_home_top + TIP + 0.4 * CPA_GAP])
            yield from move(arm, over, 1.2, label="over the CPA")
            yield from settle_until(arm, over, "line up over the CPA", tol=0.8 * MM, max_s=2.0)
            yield from press(arm, cpa_xy, cpa_home_top + TIP - 0.3 * MM, "push the CPA home", max_s=5.0,
                             max_force=HALF_CPA_FORCE if variant == "halfcpa" else None)
            report["cpa_step"] = world["step"]
        yield from move(arm, arm.pos + (0, 0, 0.06), 1.2, label="away")
        phase["right"] = "done"
        report["script_done_step"] = world["step"]
        yield from idle()

    def both_arms():
        left = watch() if variant != "hold" else idle()
        right = script()
        while True:
            next(left)
            next(right)
            yield

    driver = both_arms()
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
        d, tilt = task.plug_in_header(st)
        row = {"step": step, "left": phase["left"], "right": phase["right"], "score": res.info.get("score"),
               "success": res.info.get("success"), "plug_in_header_mm": [round(float(v), 2) for v in d],
               "tilt_deg": round(tilt, 2), "latch_mm": round(st["plug"].latch_mm, 3),
               "cpa_mm": round(st["plug"].cpa_mm, 3), "seal_mm": round(st["sensor"].seal_mm, 3),
               "latched": task.latched(st)}
        row["r_force"] = [round(float(v), 2) for v in tcp_force("right")]
        for side in ("right",):
            row[f"{side[0]}_cmd"] = [round(float(v), 4) for v in arms[side].pos]
            row[f"{side[0]}_meas"] = [round(float(v), 4) for v in measured_tcp(side)[0]]
            row[f"{side[0]}_ik"] = [round(float(v), 4) for v in getattr(arms[side], "ik_err", (0.0, 0.0))]
        log.append(row)
        if video:
            for c in CAMERAS:
                frames[c].append(res.observation.images[c])
        done_script = "script_done_step" in report and step - report["script_done_step"] >= round(2.0 / DT)
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
    d, tilt = task.plug_in_header(st)
    summary = {
        "variant": variant,
        "task_score": 1.0 if success else float(result.info.get("score", 0.0)),
        "terminated": bool(result.terminated),
        "termination_reason": result.termination_reason,
        "steps": len(log),
        "run_limit_steps": MAX_STEPS,
        "plug_in_header_mm": [float(v) for v in d],
        "seat_gap_mm": task.seat_gap_mm(st),
        "latch_mm": st["plug"].latch_mm,
        "cpa_mm": st["plug"].cpa_mm,
        "latched": task.latched(st),
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


STOP_AFTER_INSERT = False  # for mechanism tests only (see README): the arm lets go of the plug in the mouth and leaves
HALF_CPA_FORCE = 25.0  # halfcpa: the CPA pushed with up to this force (N), well over the 15 N that moves it home on a latched plug
CPA_GAP = 15 * MM  # hover this far above the CPA's home before pressing (times 0.4)


def write_videos(frames: dict, out: Path, variant: str) -> None:
    """One H.264 MP4 per camera, the frames piped to ffmpeg (on PATH) as raw RGB."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not on PATH: needed to write the videos")
    for cam, imgs in frames.items():
        if not imgs:
            continue
        h, w = imgs[0].shape[:2]
        name = out / (f"demo_{cam}.mp4" if variant == "full" else f"demo_{variant}_{cam}.mp4")
        proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                 "-s", f"{w}x{h}", "-r", str(CONTROL_HZ), "-i", "-", "-c:v", "libx264",
                                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(name)],
                                stdin=subprocess.PIPE)
        for im in imgs:
            proc.stdin.write(np.ascontiguousarray(im[..., :3], dtype=np.uint8).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {name}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=VARIANTS + ("all",), default="full",
                    help="'all' runs every variant in one simulator session (reset between them)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: runs/<variant>)")
    ap.add_argument("--no-video", action="store_true", help="no videos (and no rendering)")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--device", default="cuda:0",
                    help="physics device; 'cpu' steps this one-environment scene faster (CPU PhysX), but its "
                         "results differ in detail from the GPU solver trials run on: confirm on cuda:0")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override a demo constant for an experiment (a Python literal), e.g. --set PUSH_X=0.004")
    args = ap.parse_args(argv)
    for item in args.set:
        name, value = item.split("=", 1)
        if name not in globals():
            ap.error(f"--set: no constant {name!r}")
        globals()[name] = ast.literal_eval(value)
    video = not args.no_video
    emb = make_embodiment(video, args.device)
    for variant in VARIANTS if args.variant == "all" else (args.variant,):
        out = args.out / variant if args.out and args.variant == "all" else args.out or HERE / "runs" / variant
        summary = run(variant, out, video, args.max_steps, emb)
        print("RESULT " + json.dumps({k: summary[k] for k in ("variant", "task_score", "steps", "termination_reason",
                                                               "seat_gap_mm", "latch_mm", "cpa_mm", "wall_s")},
                                     default=float))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
