"""On the station table, in front of the arms: a loose steel TS35 x 7.5 top-hat DIN rail, 50 cm long, lying flat
(open side up) and running left to right, with an end bracket 4 cm in from each end. The rail is loose and light
(200 g): pulled or pushed on without being held, it lifts, slides and turns.

On the rail, latched, three terminal blocks about 28 mm apart: beige, blue, beige from the left. In front of the
right arm, nearer the arms than the rail: an open, shallow black parts tray (low walls, no slots), empty.

A terminal block is a push-in feed-through terminal block: 5.1 mm thick (along the rail), 65 mm long
(across it), 36 mm tall (29 mm above the rail). Its base hooks under the rail's lips: at the far end (away from the
arms) a fixed hook, which gives about 2.5 mm when pushed firmly; at the near end (toward the arms) a sprung foot. It
cannot be pulled straight off. To take it off, push it about 2.5 mm toward the arms (about 3 N; the fixed hook gives)
so the sprung foot comes out from under its lip; tip the near end up a few degrees; slide the block about 5 mm toward
the far side so the hook comes out from under its lip; lift it off.

Task: take the blue block off the rail and put it in the tray, letting go of it there. Hold the rail down with one arm
while the other pulls the block off; leave the two beige blocks latched where they are.
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

tp.RAIL_BEIGE = False  # the beige blocks here are real (latched) blocks, not part of the rail
# a longer rail than the other tasks', as cut from stock for a panel: 50 cm, ~0.33 kg/m plus two end brackets
tp.RAIL_LENGTH = 0.50
tp.RAIL_MASS = 0.20
tp.LEFT_BRACKET_Y = (0.200, 0.208)
tp.RIGHT_BRACKET_Y = (-0.208, -0.200)
tp.BLOCK_MODEL = "HOOK"  # push-in block: fixed hook at +x (far side), sprung foot at -x (near side)
tp.FIXED_HOOK = True
tp.HOOK_FLEX = 2.5 * 1e-3  # the fixed hook gives up to 2.5 mm under a firm push (the plastic bends) ...
tp.HOOK_SPRING = 1000.0  # ... at 1 N/mm (a simplification: about what the simulated arm can push sideways)
tp.BLOCK_HEIGHT = (36.15 - 7.05) * 1e-3  # the block's top above the lip tops
tp.TOUCH_WATCH.add("blue")  # the blue block's contact with the robot is measured (released: no contact)
tp.install()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"

MM = 1e-3
RAIL_ORIGIN = (0.30, -0.04, TOP + 0.2 * MM)
TRAY_ORIGIN = (0.18, -0.22, TOP)  # nearer the arms than the rail, in front of the right arm
W = tp.BLOCK_WIDTH
SPACING = W + 28 * MM  # centre to centre along the rail
MIDDLE = -0.02  # the blue block's centre along the rail (rail frame y, m)
START = {"beige_1": MIDDLE + SPACING, "blue": MIDDLE, "beige_2": MIDDLE - SPACING}
KINDS = {"beige_1": "block_beige", "blue": "block", "beige_2": "block_beige"}
BLOCKS = tuple(START)
TARGET = "blue"
SEATED_Z = RAIL_ORIGIN[2] + tp.RAIL_HEIGHT + tp.SEATED_HEIGHT

OBJECTS = [tp.part("rail", "rail", RAIL_ORIGIN), tp.part("tray", "tray", TRAY_ORIGIN)]
OBJECTS += [tp.part(n, KINDS[n], (RAIL_ORIGIN[0], RAIL_ORIGIN[1] + y, SEATED_Z), latched=True) for n, y in START.items()]

# -- scoring thresholds ----------------------------------------------------------------------------------------
RAIL_MAX_SHIFT = 10 * MM  # the rail's centre may move less than this from where it started ...
RAIL_MAX_TURN_DEG = 3.0  # ... and turn less than this
SEAT_RANGE = (-0.5 * MM, 0.8 * MM)
FEET_IN = 1.0 * MM
PADS_SQUEEZED = 0.5 * MM
LATCH_MAX_ACROSS = 1.0 * MM
MAX_SQUARE_DEG = 3.0
OFF_RAIL = 20 * MM  # a block whose base is this far above the lips (or off the rail's footprint) is off the rail
UNDISTURBED = 1.0 * MM  # a beige block moved less than this along the rail and stayed latched
TRAY_WALL_TOP = TOP + tp.TRAY_FLOOR + tp.TRAY_WALL_HEIGHT
TRAY_FLOOR_TOP = TOP + tp.TRAY_FLOOR
ON_FLOOR = 2 * MM  # "on the tray floor": the block's lowest point within this of the floor's top
RELEASED_N = 0.05  # "released": the robot presses on the block with less than this (N)
CREDIT = {"unlatched": 0.25, "off": 0.5, "in_tray": 0.75}
CAP = CREDIT["unlatched"]  # a beige block disturbed, or the rail moved

C_RAIL = np.array(tp.local_center("rail"))
C_BLOCK = np.array(tp.local_center("block"))


def _origin(state, center) -> tuple[np.ndarray, np.ndarray]:
    rot = quat_xyzw_to_matrix(np.asarray(state.quat_xyzw, dtype=float))
    return np.asarray(state.pos, dtype=float) - rot @ center, rot


def rail_pose(objects) -> tuple[np.ndarray, np.ndarray]:
    return _origin(objects["rail"], C_RAIL)


def rail_motion(objects) -> tuple[float, float]:
    """(metres the rail's origin moved across the table, degrees it turned) since the start."""
    p, rot = rail_pose(objects)
    shift = float(np.hypot(p[0] - RAIL_ORIGIN[0], p[1] - RAIL_ORIGIN[1]))
    turn = math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(rot) - 1.0) / 2.0))))
    return shift, turn


def rail_ok(objects) -> bool:
    shift, turn = rail_motion(objects)
    return shift < RAIL_MAX_SHIFT and turn < RAIL_MAX_TURN_DEG


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


def in_tray(state) -> bool:
    """Resting inside the tray: its centre inside the tray's inner walls and its lowest point below their tops."""
    lo, hi = np.asarray(state.low), np.asarray(state.high)
    c = (lo + hi) / 2
    hx, hy = tp.TRAY_INNER[0] / 2, tp.TRAY_INNER[1] / 2
    return bool(abs(c[0] - TRAY_ORIGIN[0]) <= hx and abs(c[1] - TRAY_ORIGIN[1]) <= hy and lo[2] <= TRAY_WALL_TOP)


def on_tray_floor(state) -> bool:
    """In the tray and resting on its floor: its lowest point within 2 mm of the floor's top."""
    return bool(in_tray(state) and float(np.asarray(state.low)[2]) <= TRAY_FLOOR_TOP + ON_FLOOR)


def released(state) -> bool:
    """No robot link touching the block (contact force under 0.05 N)."""
    return bool(state.robot_contact_n < RELEASED_N)


def undisturbed(objects) -> bool:
    """Both beige blocks latched within 1 mm of where they started along the rail."""
    for n in ("beige_1", "beige_2"):
        b = block_on_rail(objects, n)
        if not latched(b) or abs(b["along"] - START[n]) > UNDISTURBED:
            return False
    return True


def stage(objects) -> str | None:
    if in_tray(objects[TARGET]):
        return "in_tray"
    b = block_on_rail(objects, TARGET)
    if off_rail(b):
        return "off"
    if not latched(b):
        return "unlatched"
    return None


def success(objects) -> bool:
    """The blue block released (no robot contact) and resting still on the tray's floor, for 1 s in a row; the beige
    blocks latched where they were; the rail in place."""
    s = objects[TARGET]
    return bool(on_tray_floor(s) and released(s) and s.resting() and undisturbed(objects) and rail_ok(objects))


def score(objects) -> float:
    """Partial credit at the end of the episode (success, which needs the block released on the tray floor for 1 s,
    scores 1.0): 0.75 the blue block in the tray (released or not), 0.5 off the rail, 0.25 unlatched (pushed off-centre, its sprung foot out, or the block tipped up); capped at 0.25 if a beige block was moved 1 mm
    or unlatched, or the rail moved 1 cm or turned 3 degrees."""
    credit = CREDIT.get(stage(objects), 0.0)
    return credit if undisturbed(objects) and rail_ok(objects) else min(credit, CAP)


def distance(objects) -> float:
    """Metres from the blue block's centre to the tray's centre."""
    s = objects[TARGET]
    c = (np.asarray(s.low) + np.asarray(s.high)) / 2
    return float(np.linalg.norm(c[:2] - np.asarray(TRAY_ORIGIN[:2])))
