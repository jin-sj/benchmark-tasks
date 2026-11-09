"""On the station table, in front of the arms: a steel TS35 x 7.5 top-hat DIN rail, 50 cm long, screwed down flat
(open side up) and running left to right, with an end bracket 4 cm in from each end. On the rail, side by side with
0.2 mm between them, five identical beige terminal blocks, the middle one on the rail's centre.

A terminal block is shaped like an A3C 2.5 push-in feed-through terminal: 5.1 mm thick (along the rail), 65 mm long
(across it), 36 mm tall (29 mm above the rail). Its base hooks under the rail's lips: at the far end (away from the
arms) a fixed hook; at the near end (toward the arms) a sprung foot, which snaps under its lip when the near end is
pressed down (about 2-3 N). A latched block sits flat on the lips, square to the rail.

Task: one block is not latched. Find it and latch it, without unlatching or moving the others.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tb_parts as tp  # noqa: E402

from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_arena.urdf import quat_xyzw_to_matrix  # noqa: E402

tp.RAIL_BEIGE = False  # every block here is a real (jointed) block
tp.RAIL_LENGTH = 0.50
tp.LEFT_BRACKET_Y = (0.200, 0.208)
tp.RIGHT_BRACKET_Y = (-0.208, -0.200)
tp.BLOCK_MODEL = "A3C"  # fixed hook at +x (far side), sprung foot at -x (near side)
tp.FIXED_HOOK = True
tp.HOOK_FLEX = 2.5 * 1e-3
tp.HOOK_SPRING = 1000.0
tp.BLOCK_HEIGHT = (36.15 - 7.05) * 1e-3
tp.install()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"

MM = 1e-3
RAIL_ORIGIN = (0.30, -0.04, TOP + 0.2 * MM)
W = tp.BLOCK_WIDTH
PITCH = W + 0.2 * MM  # centre to centre along the rail
N_BLOCKS = 5
FAULTY = 4  # counted from the left (rail +y is the robot's left)
FIRST = 2 * PITCH  # the leftmost block's centre along the rail (rail frame y): the strip is centred on the rail
START = {f"block_{i}": FIRST - (i - 1) * PITCH for i in range(1, N_BLOCKS + 1)}
BLOCKS = tuple(START)
TARGET = f"block_{FAULTY}"
OTHERS = tuple(n for n in BLOCKS if n != TARGET)
SEATED_Z = RAIL_ORIGIN[2] + tp.RAIL_HEIGHT + tp.SEATED_HEIGHT

# the faulty block (not named in the docstring, which agents read): hooked on but never snapped home. Its fixed hook
# is under the far lip, its sprung foot rests on top of the near lip: the near end stands about 5 mm high, the block
# tipped about 7.5 degrees. Spawned tipped about its hook tooth's inner corner under the far lip (block frame, seated) until the
# sprung foot's lead-in rests on the near lip's top edge
HOOK_PIVOT = np.array([tp.LIP_EDGE - tp.A3C_HOOK_REACH * MM, 0.0, -tp.LIP_THICK])
RISE = tp.RESTING_BASE_HEIGHT - tp.SEATED_HEIGHT + 0.3 * MM  # the near end's base above its seated height (+ a margin)
TILT0 = math.asin(RISE / (HOOK_PIVOT[0] + tp.LIP_EDGE))


def _tilted_origin(y: float) -> tuple[float, float, float]:
    c, s = math.cos(TILT0), math.sin(TILT0)
    ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    o = HOOK_PIVOT + ry @ (-HOOK_PIVOT)  # the block's origin after tipping about the pivot
    return (float(RAIL_ORIGIN[0] + o[0]), float(RAIL_ORIGIN[1] + y), float(SEATED_Z + o[2]))


OBJECTS = [tp.part("rail", "rail", RAIL_ORIGIN, fixed=True)]
for _n, _y in START.items():
    if _n == TARGET:
        OBJECTS.append(tp.part(_n, "block_beige", _tilted_origin(_y), tilt=float(TILT0)))
    else:
        OBJECTS.append(tp.part(_n, "block_beige", (RAIL_ORIGIN[0], RAIL_ORIGIN[1] + _y, SEATED_Z), latched=True))

# -- scoring thresholds ----------------------------------------------------------------------------------------
SEAT_RANGE = (-0.5 * MM, 0.8 * MM)
FEET_IN = 1.0 * MM
PADS_SQUEEZED = 0.5 * MM
LATCH_MAX_ACROSS = 1.0 * MM
MAX_SQUARE_DEG = 3.0
OFF_RAIL = 20 * MM
UNDISTURBED = 1.0 * MM  # another block moved less than this along the rail and stayed latched
LOWERED_DEG = 5.0  # "lowered": the faulty block tipped under this (it starts at about 7.5) ...
SEATED_DEG = 3.0  # ... "seated": tipped less than this (nearly flat), but not latched
CREDIT = {"seated": 0.5, "lowered": 0.25}
CAP = 0.25  # another block unlatched or moved

C_RAIL = np.array(tp.local_center("rail"))
C_BLOCK = np.array(tp.local_center("block"))


def _origin(state, center) -> tuple[np.ndarray, np.ndarray]:
    rot = quat_xyzw_to_matrix(np.asarray(state.quat_xyzw, dtype=float))
    return np.asarray(state.pos, dtype=float) - rot @ center, rot


def rail_pose(objects) -> tuple[np.ndarray, np.ndarray]:
    return _origin(objects["rail"], C_RAIL)


def block_on_rail(objects, name: str) -> dict:
    """A block's pose in the rail's frame: across/along offsets, its base's height above the lips (at its centre and
    at each lip edge), its feet's travel and seat pads' squeeze, its yaw and tilt relative to the rail."""
    p_r, r_r = rail_pose(objects)
    p_b, r_b = _origin(objects[name], C_BLOCK)
    rel, pos = r_r.T @ r_b, r_r.T @ (p_b - p_r)
    feet = [(pos + rel @ np.array((s * tp.LIP_EDGE, 0.0, 0.0)))[2] - tp.RAIL_HEIGHT for s in (1, -1)]
    return {
        "across": float(pos[0]),
        "along": float(pos[1]),
        "height": float(pos[2] - tp.RAIL_HEIGHT),
        "feet": [float(f) for f in feet],
        "feet_spread": [float(abs(t)) for t in getattr(objects[name], "feet_spread", (math.nan, math.nan))],
        "pads_squeeze": [float(t) for t in getattr(objects[name], "pads_squeeze", (math.nan, math.nan))],
        "yaw_deg": math.degrees(math.atan2(rel[1, 0], rel[0, 0])),
        "tilt_deg": math.degrees(math.acos(min(1.0, max(-1.0, rel[2, 2])))),
    }


def latched(b) -> bool:
    """Both feet under the lips: seated flat on the lips, centred and square, the sprung foot in and both seat pads
    squeezed by the lips."""
    return bool(
        abs(b["across"]) <= LATCH_MAX_ACROSS
        and all(SEAT_RANGE[0] <= f <= SEAT_RANGE[1] for f in b["feet"])
        and all(t <= FEET_IN for t in b["feet_spread"])
        and all(t >= PADS_SQUEEZED for t in b["pads_squeeze"])
        and abs(b["yaw_deg"]) <= MAX_SQUARE_DEG
        and b["tilt_deg"] <= MAX_SQUARE_DEG
    )


def off_rail(b) -> bool:
    """Taken off: its base 20 mm or more above the lips, or its centre outside the rail's footprint."""
    return b["height"] >= OFF_RAIL or abs(b["across"]) > tp.LIP_EDGE + 30 * MM or abs(b["along"]) > tp.RAIL_LENGTH / 2


def undisturbed(objects) -> bool:
    """The four other blocks latched within 1 mm of where they started along the rail."""
    for n in OTHERS:
        b = block_on_rail(objects, n)
        if not latched(b) or abs(b["along"] - START[n]) > UNDISTURBED:
            return False
    return True


def stage(objects) -> str | None:
    """The faulty block: "latched"; "seated" (tipped under 3 degrees, not latched); "lowered" (tipped under 5
    degrees; it starts at about 7.5); "off" the rail; or None (as it started, or tipped further)."""
    b = block_on_rail(objects, TARGET)
    if off_rail(b):
        return "off"
    if latched(b):
        return "latched"
    if b["tilt_deg"] < SEATED_DEG:
        return "seated"
    if b["tilt_deg"] < LOWERED_DEG:
        return "lowered"
    return None


def success(objects) -> bool:
    """The faulty block latched and still (held 1 s), the other four latched where they were."""
    return bool(stage(objects) == "latched" and objects[TARGET].resting() and undisturbed(objects))


def score(objects) -> float:
    """Partial credit at the end of the episode (success scores 1.0): 1.0 latched, 0.5 seated (nearly flat, not
    latched), 0.25 lowered; capped at 0.25 if another block was unlatched or moved 1 mm along the rail."""
    st = stage(objects)
    credit = 1.0 if st == "latched" else CREDIT.get(st, 0.0)
    return credit if undisturbed(objects) else min(credit, CAP)


def distance(objects) -> float:
    """Degrees the faulty block is still tipped."""
    return block_on_rail(objects, TARGET)["tilt_deg"]
