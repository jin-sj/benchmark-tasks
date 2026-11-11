"""On the station table, in front of the arms: a steel TS35 x 7.5 top-hat DIN rail, 50 cm long, screwed down flat
(open side up) and running left to right, with an end bracket 4 cm in from each end. On the rail, latched, five
identical beige terminal blocks, 33 mm apart centre to centre (28 mm gaps), the middle one on the rail's centre.

A terminal block is shaped like an A3C 2.5 push-in feed-through terminal: 5.1 mm thick (along the rail), 65 mm long
(across it), 36 mm tall (29 mm above the rail). It is not symmetric: its base has a fixed hook at one end and a sprung
foot at the other, and its two ends differ in shape. Mounted the right way round, its fixed hook is under the far lip
(away from the arms) and its sprung foot under the near lip (toward the arms).

To take a block off: push it about 2.5 mm toward its sprung foot's end (about 3 N; the fixed hook gives) so the
sprung foot comes out from under its lip; tip the foot's end up a few degrees; slide the block about 5 mm toward the
hook's end so the hook comes out from under its lip; lift it off. To mount one: hook the fixed hook under the far lip
with the block tipped, its near end up; lower the near end and press it down (about 2-3 N) until the sprung foot
snaps under the near lip.

Task: one block is mounted the wrong way round. Take it off and mount it the right way round in the same place,
without disturbing the others.
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
tp.BLOCK_MODEL = "A3C"  # fixed hook at +x, sprung foot at -x (block frame)
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
RAIL_ORIGIN = (0.14, -0.066, TOP + 0.2 * MM)  # the reversed (leftmost) block on the arms' midline, 14 cm out: more
# than 60 degrees left of the right arm's base, where its wrist can still turn a block 180 degrees
W = tp.BLOCK_WIDTH
PITCH = W + 28 * MM  # centre to centre along the rail
N_BLOCKS = 5
REVERSED = 1  # counted from the left (rail +y is the robot's left); not named in the docstring, which agents read
START = {f"block_{i}": (3 - i) * PITCH for i in range(1, N_BLOCKS + 1)}
BLOCKS = tuple(START)
TARGET = f"block_{REVERSED}"
OTHERS = tuple(n for n in BLOCKS if n != TARGET)
SEATED_Z = RAIL_ORIGIN[2] + tp.RAIL_HEIGHT + tp.SEATED_HEIGHT

OBJECTS = [tp.part("rail", "rail", RAIL_ORIGIN, fixed=True)]
OBJECTS += [tp.part(n, "block_beige", (RAIL_ORIGIN[0], RAIL_ORIGIN[1] + y, SEATED_Z), latched=True,
                    yaw=math.pi if n == TARGET else 0.0) for n, y in START.items()]

# -- scoring thresholds ----------------------------------------------------------------------------------------
SEAT_RANGE = (tp.SEATED_HEIGHT - 0.8 * MM, tp.SEATED_HEIGHT + 0.5 * MM)  # 0.0 .. 1.3 mm above the lip tops
FEET_IN = 1.0 * MM
PADS_SQUEEZED = 0.5 * MM
LATCH_MAX_ACROSS = 1.0 * MM
MAX_SQUARE_DEG = 3.0
OFF_RAIL = 20 * MM
IN_PLACE = 3.0 * MM  # remounted within this of where it started along the rail (a tenth of the 28 mm gap)
UNDISTURBED = 1.0 * MM  # another block moved less than this along the rail and stayed latched
CREDIT = {"fixed": 1.0, "elsewhere": 0.75, "turned": 0.5, "off": 0.5, "loosened": 0.25}
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
    at each lip edge), its feet's travel and seat pads' squeeze, its yaw (0: the right way round, 180: reversed) and
    tilt relative to the rail."""
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


def seated(b) -> bool:
    """Both feet under the lips, either way round: seated flat on the lips, centred and square (yaw within 3 degrees
    of 0 or of 180), the sprung foot in and both seat pads squeezed by the lips."""
    yaw = abs(b["yaw_deg"])
    return bool(
        abs(b["across"]) <= LATCH_MAX_ACROSS
        and all(SEAT_RANGE[0] <= f <= SEAT_RANGE[1] for f in b["feet"])
        and all(t <= FEET_IN for t in b["feet_spread"])
        and all(t >= PADS_SQUEEZED for t in b["pads_squeeze"])
        and min(yaw, 180.0 - yaw) <= MAX_SQUARE_DEG
        and b["tilt_deg"] <= MAX_SQUARE_DEG
    )


def right_way(b) -> bool:
    """Facing the right way round: its fixed hook toward the far side (yaw within 90 degrees of 0)."""
    return abs(b["yaw_deg"]) < 90.0


def latched(b) -> bool:
    """Seated and the right way round (yaw within 3 degrees of 0)."""
    return seated(b) and abs(b["yaw_deg"]) <= MAX_SQUARE_DEG


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
    """The reversed block: "fixed" (latched the right way round, within 3 mm of its place along the rail);
    "elsewhere" (latched the right way round, not in its place); "turned" (on the rail the right way round, not
    latched); "off" the rail; "loosened" (still the wrong way round, no longer seated); None (as it started)."""
    b = block_on_rail(objects, TARGET)
    if off_rail(b):
        return "off"
    if latched(b):
        return "fixed" if abs(b["along"] - START[TARGET]) <= IN_PLACE else "elsewhere"
    if right_way(b):
        return "turned"
    return None if seated(b) else "loosened"


def success(objects) -> bool:
    """The block latched the right way round in its place and still (held 1 s), the others undisturbed."""
    return bool(stage(objects) == "fixed" and objects[TARGET].resting() and undisturbed(objects))


def score(objects) -> float:
    """Partial credit at the end of the episode (success scores 1.0): 1.0 latched the right way round in its place;
    0.75 latched the right way round elsewhere on the rail; 0.5 off the rail, or on it the right way round but not
    latched; 0.25 still the wrong way round but no longer seated; 0 as it started. Capped at 0.25 if another block
    was unlatched or moved 1 mm or more along the rail."""
    credit = CREDIT.get(stage(objects), 0.0)
    return credit if undisturbed(objects) else min(credit, CAP)


def distance(objects) -> float:
    """Degrees the block is turned from the right way round."""
    return abs(block_on_rail(objects, TARGET)["yaw_deg"])
