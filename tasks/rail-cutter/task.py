"""On the station table, in front of the arms: a powered bench DIN-rail cutter, fixed to the table, its body blue. A
DIN rail runs through it left to right: an infeed ledge on the left (680 mm long, with two low guide fences along it
that keep the rail straight), a die in the middle (a 37 x 9.5 mm opening the rail fits through, flared at its mouth)
under the cutting head (a dark grey housing with a yellow guard band; the blade inside it), and an outfeed ledge on
the right. On the outfeed ledge's top, on the side nearer the arms, a length scale measured from the blade: a mark
every 10 mm, longer marks every 50 mm and the longest every 100 mm, to 300 mm. There is no length stop.

On the infeed side, 45 mm from the blade, a toggle hold-down clamp: a steel arm hinged behind the rail, 205 mm long,
with a red handle toward the arms and a rubber pad over the rail; it rests open, raised at 45 degrees. Pushing its
handle down past 20 degrees snaps it shut, its pad holding the rail down; lifting it back opens it.

At the front, on the outfeed side, a console with a red cut button on top. Pressing the button (3 mm in) runs one
cutting stroke: the blade comes down through the die in 1 s and goes back up. The cutter only cuts with the clamp
shut; with it open the button does nothing.

A loose steel TS35 x 7.5 top-hat DIN rail, 600 mm long, open side up, lies on the infeed ledge between the fences,
its right end 40 mm short of the blade.

Task: cut a 250 mm length off the right end of the rail.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tb_parts as tp  # noqa: E402
import cutter as ct  # noqa: E402

from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_arena.urdf import quat_xyzw_to_matrix  # noqa: E402

tp.RAIL_BEIGE = False
tp.RAIL_BRACKETS = False
tp.RAIL_LENGTH = 0.60
tp.RAIL_MASS = 0.20  # 600 mm of 1 mm steel rail
ct.install()
tp.install()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 0.0
SCORE_OVER = "end"

MM = 1e-3
CUTTER_ORIGIN = (0.32, -0.04, TOP)
START_GAP = 40 * MM  # the rail's right end starts this far short of the blade
RAIL_ORIGIN = (CUTTER_ORIGIN[0], CUTTER_ORIGIN[1] + START_GAP + tp.RAIL_LENGTH / 2, TOP + ct.LEDGE_TOP)

OBJECTS = [ct.part("cutter", CUTTER_ORIGIN), tp.part("rail", "rail", RAIL_ORIGIN)]

# -- scoring --------------------------------------------------------------------------------------------------------
TARGET = 250 * MM  # the length to cut
EXACT = 1 * MM  # full credit: within this of the target ...
STILL = 0.5 * MM  # ... and the rail moved less than this along the cutter during the stroke
CLOSE = 5 * MM  # partial credit: within this
SET_S = 1.0  # set: the rail's end kept within CLOSE of the mark, in the die, this long without a break
CONTROL_HZ = 20.0  # the record runs once per control step
C_RAIL = np.array(tp.local_center("rail"))
RECORD: dict = {}


def _rail_end(objects) -> tuple[float, bool]:
    """(how far the rail's right end is past the blade (m), whether the rail lies in the die), in the cutter frame."""
    st = objects["rail"]
    rot = quat_xyzw_to_matrix(np.asarray(st.quat_xyzw, dtype=float))
    origin = np.asarray(st.pos, dtype=float) - rot @ C_RAIL  # the rail's origin: the middle of its underside
    ends = [origin + rot @ np.array([0.0, s * tp.RAIL_LENGTH / 2, 0.0]) for s in (-1.0, 1.0)]
    end = min(ends, key=lambda e: e[1])  # the end on the outfeed side (whichever way round the rail lies)
    c = np.asarray(CUTTER_ORIGIN, dtype=float)
    past = float(c[1] - end[1])  # +: past the blade toward the outfeed
    # in the die: the rail runs along the cutter (within 3 degrees), on the die's floor, centred in its opening,
    # and spans the blade's plane
    along = abs(rot[1, 1]) >= math.cos(math.radians(3.0))
    centred = abs(origin[0] - c[0]) <= ct.APERTURE[0] / 2 - 17.5 * MM + 0.5 * MM
    on_floor = abs(origin[2] - (c[2] + ct.LEDGE_TOP)) <= 2 * MM
    spans = past > 0 and past < tp.RAIL_LENGTH
    return past, bool(along and centred and on_floor and spans)


def _record(base, states) -> None:
    step = int(base.episode_length_buf[0])
    if not RECORD or step <= 1 or step < RECORD.get("step", 0):
        RECORD.clear()
        RECORD.update(cut=None, best=None, set_run=0, set_ok=False, during=None, stroke=0, step=step)
    RECORD["step"] = step
    if "rail" not in states or "cutter" not in states:
        return
    past, in_die = _rail_end(states)
    c = states["cutter"]
    if RECORD["cut"] is None:
        err = abs(past - TARGET)
        if in_die:
            RECORD["best"] = err if RECORD["best"] is None else min(RECORD["best"], err)
        RECORD["set_run"] = RECORD["set_run"] + 1 if in_die and err <= CLOSE else 0
        RECORD["set_ok"] = RECORD["set_ok"] or RECORD["set_run"] >= round(SET_S * CONTROL_HZ)
        if c.strokes > RECORD["stroke"]:  # a stroke has started: track the rail's movement through it
            RECORD["stroke"], RECORD["during"] = c.strokes, (past, past)
        if RECORD["during"] is not None:
            lo, hi = RECORD["during"]
            RECORD["during"] = (min(lo, past), max(hi, past))
            if c.blade_mm >= ct.STROKE / MM - 1.0:  # the blade at the bottom of its stroke: the cut
                lo, hi = RECORD["during"]
                RECORD["cut"] = {"in_die": in_die, "length": past, "moved": hi - lo, "step": step}


tp.STEP_HOOKS.append(_record)


def cut() -> dict | None:
    """The first stroke of the blade: whether the rail was in the die, the cut length, and how far the rail moved
    along the cutter during the stroke (from the button's starting it to the blade's reaching the bottom)."""
    return RECORD.get("cut")


def success(objects) -> bool:
    """Cut 250 mm, within 1 mm, the rail held still (moved under 0.5 mm) while the blade went through it."""
    c = cut()
    return bool(c and c["in_die"] and abs(c["length"] - TARGET) <= EXACT and c["moved"] < STILL)


def score(objects) -> float:
    """At the end of the episode: 1.0 cut 250 mm within 1 mm with the rail held still (moved under 0.5 mm) while the
    blade went through it; 0.5 cut within 5 mm; 0.25 no such cut, but before the cut the rail lay in the die with its
    end within 5 mm of the 250 mm mark for 1 s without a break (set at the mark); 0 otherwise. Only the first full
    stroke of the blade counts (a cut cannot be redone)."""
    if success(objects):
        return 1.0
    c = cut()
    if c and c["in_die"] and abs(c["length"] - TARGET) <= CLOSE:
        return 0.5
    if RECORD.get("set_ok"):
        return 0.25
    return 0.0


def distance(objects) -> float:
    """Metres from the rail's right end to the 250 mm mark."""
    return abs(_rail_end(objects)[0] - TARGET)
