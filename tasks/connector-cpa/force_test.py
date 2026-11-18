"""Mechanism test (not a demo): how hard the plug must be pushed to latch, and whether a half-mated plug stays put.

The demo's right arm lowers the plug into the shroud's mouth, lets go and leaves (``demo.STOP_AFTER_INSERT``). Then a
known force is applied straight down the mating axis to the plug's body (an external force, for this test only), by
a schedule, and the plug's height, latch arm and CPA are logged every control step:

    python force_test.py ramp      # 0 N for 1 s, then rising 2 N/s to 50 N: the force at which it latches
    python force_test.py hold25    # 25 N for 4 s, then released for 17.5 s: stays unlatched, does not back out
    python force_test.py --device cuda:0 ramp hold25

Writes ``runs/force_test/<case>.jsonl`` and prints one ``CASE`` line per case.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import demo  # noqa: E402
import tb_parts as tp  # noqa: E402

START_STEP = 400  # the schedule starts here (the arm has let go of the plug by about step 330 and left)
STATE: dict = {"t0": None, "sched": None, "log": []}


def ramp(t: float) -> float:
    return 0.0 if t < 1.0 else min(50.0, 2.0 * (t - 1.0))


def hold(force: float):
    return lambda t: force if 1.0 <= t < 5.0 else 0.0


def _hook(base, states) -> None:
    import torch

    step = int(base.episode_length_buf[0])
    if "plug" not in states or step < START_STEP:
        return
    if STATE["t0"] is None:
        STATE["t0"] = step
    t = (step - STATE["t0"]) / demo.CONTROL_HZ
    force = STATE["sched"](t)
    art = base.scene.articulations["plug"]
    f = torch.tensor([[[0.0, 0.0, -force]]], dtype=torch.float32, device=art.device)
    art.permanent_wrench_composer.set_forces_and_torques_index(forces=f, torques=torch.zeros_like(f),
                                                               body_ids=[list(art.body_names).index("plug")],
                                                               is_global=True)
    p = states["plug"]
    STATE["log"].append({"t": round(t, 2), "force_n": force, "latch_mm": round(p.latch_mm, 3),
                         "cpa_mm": round(p.cpa_mm, 3), "origin": list(p.origin), "latched": None})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*", default=["ramp", "hold25"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    demo.STOP_AFTER_INSERT = True
    tp.STEP_HOOKS.append(_hook)
    emb = demo.make_embodiment(False, args.device)
    out = HERE / "runs" / "force_test"
    out.mkdir(parents=True, exist_ok=True)
    for case in args.cases:
        STATE.update(t0=None, log=[], sched=ramp if case == "ramp" else hold(float(case[4:])))
        duration = 60.0 if case == "ramp" else 22.5
        demo.run("nocpa", out / case, False, START_STEP + round(duration * demo.CONTROL_HZ), emb)
        task = emb.task.module
        o, rot = task._header_frame()
        for r in STATE["log"]:
            d = rot.T @ (np.asarray(r.pop("origin")) - o) / 1e-3
            r["face_mm"] = round(float(d[2]), 3)
            r["off_axis_mm"] = round(float(math.hypot(d[0], d[1])), 3)
            r["latched"] = bool(math.hypot(d[0], d[1]) <= task.ON_AXIS_MM and d[2] - task.cp.SEAT <= task.SEAT_GAP_MM
                                and r["latch_mm"] <= task.LATCH_REST_MM)
        (out / f"{case}.jsonl").write_text("\n".join(json.dumps(r) for r in STATE["log"]) + "\n")
        first = next((r for r in STATE["log"] if r["latched"]), None)
        print("CASE " + json.dumps({"case": case, "latched_at": first, "end": STATE["log"][-1]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
