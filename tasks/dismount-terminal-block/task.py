"""A control panel's mounting plate on the station table, as panel builders wire it: galvanised steel, 240 x 280 x
2.5 mm, lying flat in front of the arms. On it, running left to right: a steel TS35 x 7.5 DIN rail (240 mm, screwed
down) and, 30 mm beyond the block ends on each side, a closed slotted PVC wiring duct (40 x 40 mm). On the rail between
two end brackets, a strip of 15 push-in terminal blocks: the blue one, a beige block 10 mm from each of its faces (room
for a robot fingertip, 6.3 mm thick), and beyond each of those six more beige blocks packed 0.2 mm apart. Every beige
block has a wire (1.5 mm2, 3.0 mm) in each end running straight into the duct on that side; the blue block's wires are
already disconnected. To the left, nearer the arms, beside the plate: a screwdriver holder with a flat-blade screwdriver
(4.0 x 0.8 mm blade) standing in it, handle up. In front of the right arm, beside the plate: an open black tray, empty.

A terminal block (this task's own design; every latch dimension in the README, printable parts in build/) is 6.2 mm
thick (along the rail), its housing 64.8 mm long (across it; 72.4 mm with the release tab) and 29.1 mm tall above the
rail. At the far end (away from the arms) a rigid hook (3.5 mm reach) under the far lip; at the near end a sprung foot
(2.0 mm reach, 1.0 N/mm, up to 2.5 mm out) under the near lip. With the foot under its lip, a slide toward the far side
stops after 2.8 mm, before the hook is out. The foot is released with a flat screwdriver in the release slot at the
block's near end (2.4 x 4.8 mm, 9 mm deep, between the housing's cheek and side walls and the foot's tab), levered
outward (2.0 N at 2.0 mm, where its tooth clears the lip); then the near end is tipped up, the block slid a few
millimetres toward the far side so the hook comes out, and lifted off.

Task: take the blue block off the rail with the screwdriver and put it in the tray, letting go of it there; leave the
beige blocks latched where they are (the two beside it are scored).
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
tp.RAIL_LENGTH = 0.24  # the rail on the panel's mounting plate, screwed down
tp.RAIL_MASS = 0.20
tp.BLOCK_MODEL = "HOOK"  # push-in block: fixed hook at +x (far side), sprung foot at -x (near side)
tp.FIXED_HOOK = True
tp.HOOK_FLEX = 0.0  # the fixed hook is rigid: the block is released only through the sprung foot ...
tp.RELEASE_SLOT = True  # ... with a screwdriver in the release slot at its near end
# The release slot (block frame x, mm; tb_parts.HOOK_SLOT_CHEEK, SLOT_Z, SLOT_FLOOR): the housing's cheek x -30.0..-27.8,
# the slot x -32.4..-30.0 (2.4 mm wide, 4.8 mm long along the rail, its floor at z 0, open at the top at z 9.0: 9 mm
# deep), the foot's tab upright x -35.4..-32.4 (3.0 mm thick, z -1.0..9.0) and its floor under the slot (z -1.0..0).
# A 4.0 x 0.8 mm blade (tb_parts.BLADE) goes in with 1.6 mm of play across the slot.
tp.TAB_THICK = 3.0 * 1e-3
tp.SLOT_WIDTH = 2.4 * 1e-3
# The latch's travel: the sprung foot moves out at most 2.5 mm (its tooth clears the lip at 2.0 mm), and the fixed
# hook reaches 3.5 mm under its lip. With the foot under its lip, sliding the block toward the far side stops after
# 0.3 + 2.5 = 2.8 mm, so the hook can't be slid out: the foot must be released first.
tp.FOOT_TRAVEL = 2.5 * 1e-3
tp.HOOK_REACH = 3.5  # mm
tp.CONTACT_OFFSET = 0.2e-3  # PhysX contact offset of the parts (0.5 mm let the GPU solver hold a 0.8 mm blade at the
# 2.4 mm slot's edge)
tp.BLOCK_HEIGHT = (36.15 - 7.05) * 1e-3  # the block's top above the lip tops
tp.BLOCK_WIDTH = 6.2e-3  # a 4 mm2 push-in feed-through block
tp.FOOT_HALF_WIDTH = 2.95e-3  # the latch parts 0.15 mm inside each face
tp.BLOCK_MASS = 0.0081  # housing; 9.5 g with the feet, pads, guide pin and spring
tp.TOUCH_WATCH.add("blue")  # the blue block's contact with the robot is measured (released: no contact)

MM = 1e-3
# -- the terminal strip, on the rail on the panel's mounting plate -------------------------------------------------
GAP = 0.2 * MM  # between neighbouring blocks (packed strip)
BLUE_GAP = float(__import__("os").environ.get("TB_BLUE_GAP_MM", "10.0")) * MM  # either side of the blue block: room for a robot fingertip (6.3 mm thick) beside it
PITCH = tp.BLOCK_WIDTH + GAP
N_STATIC = 6  # blocks fixed in the strip beyond each latched beige block
MIDDLE = -0.02  # the blue block's centre along the rail (rail frame y, m)
NEIGHBOUR = tp.BLOCK_WIDTH + BLUE_GAP  # the blue block's centre to each latched beige block's
START = {"beige_1": MIDDLE + NEIGHBOUR, "blue": MIDDLE, "beige_2": MIDDLE - NEIGHBOUR}
STATIC = tuple(START[n] + s * k * PITCH for n, s in (("beige_1", 1), ("beige_2", -1)) for k in range(1, N_STATIC + 1))
_ends = (max(STATIC) + tp.BLOCK_WIDTH / 2 + GAP, min(STATIC) - tp.BLOCK_WIDTH / 2 - GAP)
tp.LEFT_BRACKET_Y = (_ends[0], _ends[0] + 8 * MM)  # an end bracket against each end of the strip
tp.RIGHT_BRACKET_Y = (_ends[1] - 8 * MM, _ends[1])
tp.PANEL = True
tp.STATIC_BLOCKS = STATIC
tp.WIRED = tuple(sorted(STATIC + (START["beige_1"], START["beige_2"])))  # every beige block is wired at both ends
tp.install()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"

RAIL_ORIGIN = (0.30, -0.04, TOP + tp.PLATE[4] + 0.2 * MM)  # the rail's web on the mounting plate's top
TRAY_ORIGIN = (0.18, -0.225, TOP)  # nearer the arms than the rail, in front of the right arm, beside the plate
KINDS = {"beige_1": "block_beige", "blue": "block", "beige_2": "block_beige"}
BLOCKS = tuple(START)
TARGET = "blue"
SEATED_Z = RAIL_ORIGIN[2] + tp.RAIL_HEIGHT + tp.SEATED_HEIGHT

HOLDER_ORIGIN = (0.16, 0.08, TOP)  # the screwdriver holder's base centre: left of the rail's middle, nearer the arms
SCREWDRIVER_START = (HOLDER_ORIGIN[0], HOLDER_ORIGIN[1],  # its blade's tip; it stands in the holder on its handle
                     TOP + tp.HOLDER[2] - tp.SHAFT[1] + 0.5 * MM)
OBJECTS = [tp.part("rail", "rail", RAIL_ORIGIN, fixed=True), tp.part("tray", "tray", TRAY_ORIGIN),
           tp.part("holder", "holder", HOLDER_ORIGIN, fixed=True),
           tp.part("screwdriver", "screwdriver", SCREWDRIVER_START)]
OBJECTS += [tp.part(n, KINDS[n], (RAIL_ORIGIN[0], RAIL_ORIGIN[1] + y, SEATED_Z), latched=True) for n, y in START.items()]

# -- scoring thresholds ----------------------------------------------------------------------------------------
RAIL_MAX_SHIFT = 10 * MM  # the rail's centre may move less than this from where it started ...
RAIL_MAX_TURN_DEG = 3.0  # ... and turn less than this
SEAT_RANGE = (-0.5 * MM, 0.8 * MM)
FEET_IN = 1.0 * MM
PADS_SQUEEZED = 0.5 * MM
LATCH_MAX_ACROSS = 1.0 * MM
MAX_SQUARE_DEG = 3.0
FOOT_CLEAR = 2.0 * MM  # "foot released": its tooth pushed out this far past its latched position, measured against
# the rail (it clears the lip's edge at 2.0 mm) ...
FOOT_ABOVE = -tp.FOOT_BOTTOM  # ... or the near end lifted this far (6.5 mm): the foot's tooth above the lip's top
OFF_RAIL = 20 * MM  # a block whose base is this far above the lips (or off the rail's footprint) is off the rail
UNDISTURBED = 1.0 * MM  # a beige block moved less than this along the rail and stayed latched
TRAY_WALL_TOP = TOP + tp.TRAY_FLOOR + tp.TRAY_WALL_HEIGHT
TRAY_FLOOR_TOP = TOP + tp.TRAY_FLOOR
ON_FLOOR = 2 * MM  # "on the tray floor": the block's lowest point within this of the floor's top
RELEASED_N = 0.05  # "released": the robot presses on the block with less than this (N)
CREDIT = {"foot_released": 0.25, "off": 0.5, "in_tray": 0.75}
CAP = CREDIT["foot_released"]  # a beige block disturbed, or the rail moved

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


def foot_out(b) -> float:
    """How far the sprung (near) foot's tooth is out past where it sits latched, in the rail's frame (metres): the
    foot's travel on the block less the block's own shift toward the far side (a block slid toward the far side carries
    the foot back in under its lip)."""
    return b["feet_spread"][1] - b["across"]


def foot_released(b) -> bool:
    """The sprung foot measurably released: its tooth pushed out 2.0 mm or more past its latched position (clear of the
    lip's edge), or the block's near end lifted 6.5 mm or more (the foot's tooth above the lip's top); ``feet`` is
    (far, near)."""
    return bool(foot_out(b) >= FOOT_CLEAR or b["feet"][1] >= FOOT_ABOVE)


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
    if foot_released(b):
        return "foot_released"
    return None


def success(objects) -> bool:
    """The blue block released (no robot contact) and resting still on the tray's floor, for 1 s in a row; the beige
    blocks latched where they were; the rail in place."""
    s = objects[TARGET]
    return bool(on_tray_floor(s) and released(s) and s.resting() and undisturbed(objects) and rail_ok(objects))


def score(objects) -> float:
    """Partial credit at the end of the episode (success, which needs the block released on the tray floor for 1 s,
    scores 1.0): 0.75 the blue block in the tray (released or not), 0.5 off the rail (its base 20 mm or more above the
    lips, or its centre outside the rail's footprint), 0.25 its sprung foot released (its tooth 2.0 mm or more out past
    its latched position, measured against the rail, or the near end lifted 6.5 mm or more off the lip); 0 otherwise,
    however far a still-latched block was shifted or tilted. Capped at 0.25 if a
    beige block was moved 1 mm or unlatched, or the rail moved 1 cm or turned 3 degrees."""
    credit = CREDIT.get(stage(objects), 0.0)
    return credit if undisturbed(objects) and rail_ok(objects) else min(credit, CAP)


def distance(objects) -> float:
    """Metres from the blue block's centre to the tray's centre."""
    s = objects[TARGET]
    c = (np.asarray(s.low) + np.asarray(s.high)) / 2
    return float(np.linalg.norm(c[:2] - np.asarray(TRAY_ORIGIN[:2])))
