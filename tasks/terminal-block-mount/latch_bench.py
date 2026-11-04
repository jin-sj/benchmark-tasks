"""Latch bench (not a benchmark task): fixed DIN rails side by side on the station table, each with one blue terminal
block on it, and one loose rail. Each block is pushed, pulled or tipped by a scripted external force while the arms
stay in their ready pose, so the latch's forces can be measured under exactly the task's physics.

    python latch_bench.py [--out runs/latch_bench]

Cases (force ramps; the block's own reaction is measured, not prescribed):

- press, press_offset_0.5mm, press_offset_1mm: a block resting unpressed on the lips is pushed straight down at
  base level; the force at which both feet click under the lips. (A block left resting
  unpressed is only marginally upright on its lead-ins, and a push of several newtons on its top buckles it
  sideways; in the task the hand holds it square, so here the push acts at base level between the feet.)
- seated, seated_flipped: latched blocks left alone (the second turned 180 degrees): how they sit.
- slide: a latched block is pushed along the rail 2.5 mm above the lips (where a fingertip pushes it); the
  force at which it starts to slide, and how far it pitches before that.
- release: a latched block is pushed along the rail with 1.1x the slide force until it has moved 3 mm, then let
  go: does it stop where it was left, and settle back flat?
- pull: a latched block is pulled straight up at its centre; the pull-off force.
- tilt_off: a latched block is lifted at one end of its long wing; the force that tips it off the rail.
- rail_slide: the loose rail (no block) is pushed along its length at its centre; the force at which it slides.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tb_parts as tp  # noqa: E402

from yam_arena.task_file import TABLE_TOP  # noqa: E402

tp.install()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
MM = 1e-3
PRESS = ("press", "press_offset_05mm", "press_offset_1mm")
CASES = PRESS + ("seated", "seated_flipped", "slide", "release", "pull", "tilt_off")
RAIL_X = {c: 0.20 + 0.055 * i for i, c in enumerate(CASES)}
RAIL_Y = 0.0
# neighbouring rails are closer than a block is long, so blocks alternate along their rails
BLOCK_Y = {c: (-0.05 if i % 2 else 0.02) for i, c in enumerate(CASES)}
BLOCK_Y["release"] = -0.09  # it is pushed toward +y: keep it well away from the beige block
LOOSE_RAIL = (0.20 + 0.055 * len(CASES), RAIL_Y, TOP + 0.2 * MM)
SEATED_Z = TOP + 0.2 * MM + tp.RAIL_HEIGHT + tp.SEATED_HEIGHT


def _block_pose(case):
    x, z, yaw = RAIL_X[case], SEATED_Z, 0.0
    if case.startswith("press"):
        z += tp.RESTING_BASE_HEIGHT - tp.SEATED_HEIGHT + 0.5 * MM
    if case == "press_offset_05mm":
        x += 0.5 * MM
    if case == "press_offset_1mm":
        x += 1.0 * MM
    if case == "seated_flipped":
        yaw = math.pi
    return (x, RAIL_Y + BLOCK_Y[case], z), yaw


OBJECTS = [tp.part(f"rail_{c}", "rail", (RAIL_X[c], RAIL_Y, TOP + 0.2 * MM), fixed=True) for c in CASES]
OBJECTS += [tp.part(f"block_{c}", "block", _block_pose(c)[0], yaw=_block_pose(c)[1], latched=not c.startswith("press"))
            for c in CASES]
OBJECTS.append(tp.part("rail_loose", "rail", LOOSE_RAIL))

SETTLE_S = 1.0
RAMP_S = 5.0
PEAK = {"press": 20.0, "slide": 6.0, "pull": 40.0, "tilt_off": 30.0, "rail_slide": 4.0}  # N at the end of the ramp


def run(out: Path, device: str = "cuda:0") -> dict:
    from inspect_robots import Scene
    from inspect_robots.types import Action

    from yam_arena.inspect_embodiment import YamArenaEmbodiment

    import torch

    # control at the physics rate, so every physics step is logged (a real click shows the feet spreading past the
    # lip edge; a tooth tunnelling through the lip would not)
    emb = YamArenaEmbodiment(task=__file__, control_hz=240.0, object_jitter=0.0, cam_height=32, cam_width=32,
                             device=device)
    obs = emb.reset(Scene(id="latch-bench", instruction="bench"), seed=0)
    emb._env.unwrapped.cfg.sim.render_interval = 10**9  # no rendering: the bench reads states only
    hold = Action(data=np.asarray(obs.state["joint_pos"], dtype=float))
    base = emb._env.unwrapped
    origin = base.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
    dt = 1.0 / 240.0
    n_steps = round((SETTLE_S + RAMP_S + 1.0) / dt)

    def block_pose(case):
        art = base.scene.articulations[f"block_{case}"]
        p = art.data.root_pose_w.torch[0].detach().cpu().numpy().astype(np.float64)
        x, y, z, w = p[3:]
        rot = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        q = art.data.joint_pos.torch[0].detach().cpu().numpy()
        names = list(art.joint_names)
        spread = (float(q[names.index("foot_p")]), float(-q[names.index("foot_n")]))
        lift = (float(q[names.index("pad_p")]), float(q[names.index("pad_n")]))  # pads squeezed
        return p[:3] - origin, rot, spread, lift

    def push(case, force_world, point_local):
        """A permanent force (world frame, N) at a point given in the block's frame."""
        art = base.scene.articulations[f"block_{case}"]
        pos, rot, *_ = block_pose(case)
        point = rot @ np.asarray(point_local) + pos + origin
        bid = art.find_bodies("body")[0][0]
        f = torch.tensor([[list(force_world)]], dtype=torch.float32, device=base.device)
        p = torch.tensor([[list(point)]], dtype=torch.float32, device=base.device)
        art.permanent_wrench_composer.reset()
        art.permanent_wrench_composer.add_forces_and_torques_index(forces=f, positions=p, body_ids=[bid], is_global=True)

    def push_rail(force_world):
        rail = base.scene.rigid_objects["rail_loose"]
        f = torch.tensor([[list(force_world)]], dtype=torch.float32, device=base.device)
        p = rail.data.root_pose_w.torch[0, :3].detach().clone().reshape(1, 1, 3)
        p[..., 2] += tp.SHEET / 2
        rail.permanent_wrench_composer.reset()
        rail.permanent_wrench_composer.add_forces_and_torques_index(forces=f, positions=p, body_ids=[0], is_global=True)

    start = {c: block_pose(c) for c in CASES}
    rail0 = base.scene.rigid_objects["rail_loose"].data.root_pose_w.torch[0, :3].detach().cpu().numpy().copy()
    log = {c: [] for c in CASES + ("rail_slide",)}
    release_force = None
    for step in range(n_steps):
        t = step * dt
        ramp = min(max((t - SETTLE_S) / RAMP_S, 0.0), 1.0)
        for c in CASES:
            pos, rot, feet, lift = block_pose(c)
            rel = pos - np.array([RAIL_X[c], RAIL_Y + BLOCK_Y[c], SEATED_Z])
            pitch = math.degrees(math.atan2(rot[2, 1], rot[2, 2]))  # about the across axis (x)
            roll = math.degrees(math.atan2(rot[2, 0], rot[0, 0]))  # about the rail's axis (y)
            log[c].append({"t": round(t, 3), "x_mm": rel[0] / MM, "y_mm": rel[1] / MM, "z_mm": rel[2] / MM,
                           "pitch_deg": pitch, "roll_deg": roll, "feet_mm": [f / MM for f in feet],
                           "lift_mm": [v / MM for v in lift]})
        rail_now = base.scene.rigid_objects["rail_loose"].data.root_pose_w.torch[0, :3].detach().cpu().numpy()
        log["rail_slide"].append({"t": round(t, 3), "y_mm": (rail_now[1] - rail0[1]) / MM})
        # forces for the next step
        if t >= SETTLE_S:
            for c in PRESS:
                f = PEAK["press"] * ramp
                push(c, (0, 0, -f), (0.0, 0.0, 0.5 * MM))  # at base level, between the feet: no lever to tip it
                log[c][-1]["force_N"] = f
            f = PEAK["slide"] * ramp
            push("slide", (0, f, 0), (0.0, -tp.BLOCK_WIDTH / 2, 2.5 * MM))
            log["slide"][-1]["force_N"] = f
            for c, point in (("pull", (-2.0 * MM, 0.0, 10.0 * MM)), ("tilt_off", (24.0 * MM, 0.0, 6.0 * MM))):
                off = any(r["z_mm"] > 1.5 for r in log[c])  # once it is off the rail, stop pulling
                f = 0.0 if off else PEAK[c] * ramp
                push(c, (0, 0, f), point)
                log[c][-1]["force_N"] = f
            push_rail((0, PEAK["rail_slide"] * ramp, 0))
            log["rail_slide"][-1]["force_N"] = PEAK["rail_slide"] * ramp
            # release: push with 1.1x the slide case's breakaway once it is known (else 2 N) until it has moved 3 mm
            # (at most 1 s), then let go
            if release_force is None:
                moved = [r for r in log["slide"] if abs(r["y_mm"]) > 0.3 and "force_N" in r]
                if moved or t >= SETTLE_S + 2.0:
                    release_force = 1.1 * (moved[0]["force_N"] if moved else 2.0)
                    release_t0 = t
            pushed = [r for r in log["release"] if r.get("force_N", 0) > 0]
            far = bool(pushed) and log["release"][-1]["y_mm"] - pushed[0]["y_mm"] > 3.0
            if release_force is not None and t < release_t0 + 1.0 and not far and (not pushed or pushed[-1] is log["release"][-2]):
                push("release", (0, release_force, 0), (0.0, -tp.BLOCK_WIDTH / 2, 2.5 * MM))
                log["release"][-1]["force_N"] = release_force
            else:
                push("release", (0, 0, 0), (0.0, 0.0, 0.0))
                log["release"][-1]["force_N"] = 0.0
        emb.step(hold)

    def first(rows, cond):
        for r in rows:
            if "force_N" in r and cond(r):
                return r
        return None

    summary = {}
    for c in ("seated", "seated_flipped"):
        end = log[c][-1]
        summary[c] = {"height_mm": round(end["z_mm"], 3), "x_mm": round(end["x_mm"], 3), "roll_deg": round(end["roll_deg"], 3),
                      "pitch_deg": round(end["pitch_deg"], 3), "feet_mm": [round(f, 3) for f in end["feet_mm"]],
                      "lift_mm": [round(v, 3) for v in end["lift_mm"]]}
    for c in PRESS:
        hit = first(log[c], lambda r: abs(r["z_mm"]) < 0.3 and all(f < 1.0 for f in r["feet_mm"])
                    and all(v > 0.6 for v in r["lift_mm"]))
        end = log[c][-1]
        summary[c] = {"start_height_mm": round(log[c][round(SETTLE_S / dt) - 1]["z_mm"], 3),
                      "latch_force_N": None if hit is None else round(hit["force_N"], 2),
                      "max_spread_mm": round(max(max(r["feet_mm"]) for r in log[c]), 3),
                      "final_height_mm": round(end["z_mm"], 3), "final_feet_mm": [round(f, 3) for f in end["feet_mm"]],
                      "final_lift_mm": [round(v, 3) for v in end["lift_mm"]]}
    rows = log["slide"]
    pre = rows[round(SETTLE_S / dt) - 1]
    hit = first(rows, lambda r: abs(r["y_mm"] - pre["y_mm"]) > 0.3)
    summary["slide"] = {"seated_height_mm": round(pre["z_mm"], 3), "seated_feet_mm": [round(f, 3) for f in pre["feet_mm"]],
                        "breakaway_N": None if hit is None else round(hit["force_N"], 2),
                        "max_pitch_before_deg": round(max(abs(r["pitch_deg"]) for r in rows if hit is None or r["t"] < hit["t"]), 3)}
    rows = log["release"]
    on = [r for r in rows if r.get("force_N", 0) > 0]
    off = [r for r in rows if "force_N" in r and r["force_N"] == 0 and on and r["t"] > on[-1]["t"]]
    summary["release"] = {"push_N": None if release_force is None else round(release_force, 2),
                          "moved_while_pushed_mm": round(on[-1]["y_mm"] - on[0]["y_mm"], 3) if on else None,
                          "max_pitch_pushed_deg": round(max((abs(r["pitch_deg"]) for r in on), default=0.0), 3),
                          "drift_after_release_mm": round(off[-1]["y_mm"] - off[0]["y_mm"], 3) if off else None,
                          "pitch_after_release_deg": round(off[-1]["pitch_deg"], 3) if off else None,
                          "height_after_release_mm": round(off[-1]["z_mm"], 3) if off else None}
    for c in ("pull", "tilt_off"):
        hit = first(log[c], lambda r: r["z_mm"] > 1.5)
        prev = [r for r in log[c] if "force_N" in r and (hit is None or r["t"] < hit["t"])]
        summary[c] = {"release_force_N": None if hit is None else round(prev[-1]["force_N"], 2),
                      "max_feet_before_mm": [round(max(r["feet_mm"][i] for r in prev), 3) for i in (0, 1)] if prev else None}
    hit = first(log["rail_slide"], lambda r: abs(r["y_mm"]) > 0.3)
    summary["rail_slide"] = {"breakaway_N": None if hit is None else round(hit["force_N"], 2)}
    for c in CASES:
        summary[c]["settled_rest_mm"] = round(log[c][round(SETTLE_S / dt) - 1]["z_mm"], 3)
    out.mkdir(parents=True, exist_ok=True)
    (out / "latch_bench.json").write_text(json.dumps({"summary": summary, "log": log}, indent=1, default=float))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override a tb_parts constant (mass, spring, armature...) for an experiment")
    ap.add_argument("--device", default="cuda:0", help="physics device (cpu is faster for this scene)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs" / "latch_bench")
    args = ap.parse_args(argv)
    import ast
    for item in args.set:
        name, value = item.split("=", 1)
        if not hasattr(tp, name):
            ap.error(f"--set: tb_parts has no constant {name!r}")
        setattr(tp, name, ast.literal_eval(value))
    summary = run(args.out, args.device)
    print("BENCH " + json.dumps(summary, default=float))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
