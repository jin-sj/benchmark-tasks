"""On the station table, in front of the arms: a loose steel TS35 x 7.5 top-hat DIN rail, 24 cm long, lying flat
(open side up) and running left to right. Fixed on it: an end bracket near each end and a beige terminal block
already mounted against the left bracket. The rail is loose and light (150 g): pushed or pressed on without being
held, it slides and turns.

In front of the right arm: an open, shallow black parts tray (low walls, no slots) holding three blue terminal
blocks standing upright side by side in a staggered row, each leaning on its neighbour.

A terminal block is shaped like a WDU 2.5 feed-through terminal: 5.1 mm thick (along the rail), 60 mm long
(across it), 46 mm tall (39.5 mm above the rail), with a clamping screw on top near each end. At each end of its
base is a spring foot that clicks under the rail's lips when the block is pressed straight down onto the rail
(about 5 N); a block just set on the rail rests on its feet about 4.5 mm high and is not latched. A latched block is
clamped onto the rail: it slides along it only when pushed with about 2 N, and stays where it is left. Blocks
cannot be slid on from a rail end: the end brackets are in the way.

Task: mount one blue block next to the beige one. Hold the rail down with one arm; with the other, take a blue
block from the tray, press it straight down onto the rail until it clicks, and slide it along the rail until it
sits flush against the beige block.
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

tp.install()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"

MM = 1e-3
RAIL_ORIGIN = (0.30, -0.04, TOP + 0.2 * MM)
TRAY_ORIGIN = (0.30, -0.27, TOP)
BLOCK_BASE_Z = TOP + tp.TRAY_FLOOR - tp.FOOT_BOTTOM + 0.2 * MM
# Three blocks side by side (5.2 mm apart along y), staggered 44 mm along x so each end block's middle is free on
# both faces for a grip across them, while every block leans on its neighbour over 16 mm of its length.
BLOCK_XY = {
    "blue_1": (TRAY_ORIGIN[0] - 0.044, TRAY_ORIGIN[1] - 0.0052),
    "blue_2": (TRAY_ORIGIN[0], TRAY_ORIGIN[1]),
    "blue_3": (TRAY_ORIGIN[0] + 0.044, TRAY_ORIGIN[1] + 0.0052),
}
BLOCKS = tuple(BLOCK_XY)

OBJECTS = [
    tp.part("rail", "rail", RAIL_ORIGIN),
    tp.part("tray", "tray", TRAY_ORIGIN),
    *(tp.part(name, "block", (x, y, BLOCK_BASE_Z)) for name, (x, y) in BLOCK_XY.items()),
]

# -- scoring thresholds (all checkable with a ruler, a feeler gauge and a photo) ----------------------------
RAIL_MAX_SHIFT = 10 * MM  # the rail's centre may move less than this from where it started ...
RAIL_MAX_TURN_DEG = 3.0  # ... and turn less than this
LIFT_HEIGHT = 25 * MM  # a block counts as lifted when its lowest point is this far above the table
SEAT_RANGE = (-0.5 * MM, 0.8 * MM)  # the base above the lip tops at each lip edge: seated (latched: 0.3 mm)
FEET_IN = 1.0 * MM  # each foot spread at most this far: its tooth is under the lip (it clears the edge at 2 mm)
PADS_SQUEEZED = 0.5 * MM  # each seat pad squeezed at least this far: the lip is clamped between it and the tooth
REST_MAX = 8.0 * MM  # each foot's base at most this far above the lips: the block rests on the rail (perched on
# the lead-ins it sits ~5.4 mm up; hooked under one lip and tipped onto the other edge, ~6-7 mm at the far edge)
LATCH_MAX_ACROSS = 1.0 * MM  # the block's centre from the rail's centreline, latched
REST_MAX_ACROSS = 4.0 * MM
MAX_SQUARE_DEG = 3.0  # latched: square to the rail (yaw) and level (tilt)
REST_MAX_TILT_DEG = 10.0
FLUSH_GAP = 0.5 * MM  # gap to the beige block at each of the four check points on the face
# the four check points on the block's face toward the beige block: (across, height above the lips), at both ends of
# the base and on the two outer posts near the top (37.5 mm apart vertically: 0.5 mm there is 0.76 degrees)
FLUSH_POINTS = ((15 * MM, 0.5 * MM), (15 * MM, 38 * MM), (-15 * MM, 0.5 * MM), (-15 * MM, 38 * MM))
CREDIT = {"lifted": 0.25, "resting": 0.5, "latched": 0.75}
RAIL_MOVED_CAP = CREDIT["lifted"]

C_RAIL = np.array(tp.local_center("rail"))
C_BLOCK = np.array(tp.local_center("block"))


def _origin(state, center) -> tuple[np.ndarray, np.ndarray]:
    """(origin, rotation) of a part from its ObjectState (whose ``pos`` is its bounding-box centre)."""
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


def block_on_rail(objects, name) -> dict:
    """A block's pose in the rail's frame: across/along offsets, its base's height above the lips at each lip edge,
    its feet's travel, its yaw and tilt relative to the rail, and its gap to the beige block at the four check
    points of its facing side."""
    p_r, r_r = rail_pose(objects)
    p_b, r_b = _origin(objects[name], C_BLOCK)
    rel, pos = r_r.T @ r_b, r_r.T @ (p_b - p_r)

    def point(local):
        return pos + rel @ np.asarray(local, dtype=float)

    feet = [point((s * tp.LIP_EDGE, 0.0, 0.0))[2] - tp.RAIL_HEIGHT for s in (1, -1)]
    corners = [point((x, tp.BLOCK_WIDTH / 2, z)) for x, z in FLUSH_POINTS]
    return {
        "across": float(pos[0]),
        "along": float(pos[1]),
        "feet": [float(f) for f in feet],
        "feet_spread": [float(t) for t in getattr(objects[name], "feet_spread", (math.nan, math.nan))],
        "pads_squeeze": [float(t) for t in getattr(objects[name], "pads_squeeze", (math.nan, math.nan))],
        "yaw_deg": math.degrees(math.atan2(rel[1, 0], rel[0, 0])),
        "tilt_deg": math.degrees(math.acos(min(1.0, max(-1.0, rel[2, 2])))),
        "gaps": [float(tp.BEIGE_FACE_Y - c[1]) for c in corners],
    }


def _between_brackets(b) -> bool:
    return tp.RIGHT_BRACKET_Y[1] + tp.BLOCK_WIDTH / 2 - 1 * MM <= b["along"] <= tp.BEIGE_FACE_Y - tp.BLOCK_WIDTH / 2 + 1 * MM


def latched(b) -> bool:
    """Both feet clicked under the lips: the block seated flat on the lips, centred and square, between the
    brackets, both feet back in (a foot still spread on the lip edge has not clicked) and both seat pads squeezed
    by the lips (the clamp)."""
    return bool(
        _between_brackets(b)
        and abs(b["across"]) <= LATCH_MAX_ACROSS
        and all(SEAT_RANGE[0] <= f <= SEAT_RANGE[1] for f in b["feet"])
        and all(t <= FEET_IN for t in b["feet_spread"])
        and all(t >= PADS_SQUEEZED for t in b["pads_squeeze"])
        and abs(b["yaw_deg"]) <= MAX_SQUARE_DEG
        and b["tilt_deg"] <= MAX_SQUARE_DEG
    )


def resting(b) -> bool:
    """On the rail between the brackets, centred within 4 mm, its base within 8 mm of the lips at both edges and
    tilted at most 10 degrees: perched on the lead-ins, or hooked under one lip and tipped onto the other."""
    return bool(
        _between_brackets(b)
        and abs(b["across"]) <= REST_MAX_ACROSS
        and all(SEAT_RANGE[0] <= f <= REST_MAX for f in b["feet"])
        and b["tilt_deg"] <= REST_MAX_TILT_DEG
    )


def flush(b) -> bool:
    return latched(b) and all(-FLUSH_GAP <= g <= FLUSH_GAP for g in b["gaps"])


def stage(objects, name) -> str | None:
    b = block_on_rail(objects, name)
    if flush(b):
        return "flush"
    if latched(b):
        return "latched"
    if resting(b):
        return "resting"
    if objects[name].low[2] >= TOP + LIFT_HEIGHT:
        return "lifted"
    return None


def success(objects) -> bool:
    """A blue block latched flush against the beige block and still (held 1 s), with the rail where it started."""
    return rail_ok(objects) and any(stage(objects, n) == "flush" and objects[n].resting() for n in BLOCKS)


def score(objects) -> float:
    """Partial credit at the end of the episode (success scores 1.0): 0.75 latched (flush or not), 0.5 resting on
    the lips, 0.25 lifted; best over the blue blocks. Capped at 0.25 if the rail moved 1 cm or turned 3 degrees."""
    best = 0.0
    for name in BLOCKS:
        st = stage(objects, name)
        best = max(best, CREDIT.get("latched" if st == "flush" else st, 0.0))
    return best if rail_ok(objects) else min(best, RAIL_MOVED_CAP)


def distance(objects) -> float:
    """Metres from the nearest blue block's centre to the mounted slot, in the rail's frame."""
    out = []
    for name in BLOCKS:
        b = block_on_rail(objects, name)
        foot = max(b["feet"])
        out.append(math.sqrt(b["across"] ** 2 + (b["along"] - tp.SLOT_Y) ** 2 + max(0.0, foot) ** 2))
    return float(min(out))
