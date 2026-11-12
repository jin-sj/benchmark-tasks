"""On the station table, in front of the arms: a loose steel TS35 x 7.5 top-hat DIN rail, 24 cm long, lying flat
(open side up) and running left to right. Fixed on it: an end bracket near each end and a beige terminal block
already mounted against the left bracket. The rail is loose and light (150 g): pushed or pressed on without being
held, it slides and turns.

A person's hand and forearm (skin-coloured, in a grey sleeve) reach in over the far edge of the table, in front of
the right arm, holding a blue terminal block pinched low down between thumb and index finger by its far end, upright, its
length pointing at the robot. Over 3 s the hand brings it to about 0.35 m in front of the arms and 13 cm above the
table, and holds it there, drifting by a few millimetres as a person does. The hand lets go only once the robot is
gripping the block firmly (both fingertips, for about 0.3 s), then withdraws. It never lets go otherwise.

A terminal block is a screw-clamp feed-through terminal block: 5.1 mm thick, 60 mm long, 46 mm tall (39.5 mm
above the rail when mounted). At each end of its base is a spring foot that clicks under the rail's lips when the
block is pressed straight down onto the rail (about 5 N); a block just set on the rail rests on its feet about 4.5
mm high and is not latched.

Task: take the block from the person's hand and mount it on the rail. Grip it while the hand still holds it, so
the hand can let go; do not snatch it, drop it, press on the hand, or push or pull the block against the hand. Then press the block straight down onto the rail
anywhere between the beige block and the right bracket until it clicks. Do not knock the rail out of place: moving it
1 cm or turning it 3 degrees spoils the mount (the other arm may hold it down).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hand  # noqa: E402
import tb_parts as tp  # noqa: E402

from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_arena.urdf import quat_xyzw_to_matrix  # noqa: E402

tp.install()
hand.install()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"

MM = 1e-3
RAIL_ORIGIN = (0.30, -0.04, TOP + 0.2 * MM)
HAND_START = (0.70, -0.15, TOP + 0.20)  # the pinch centre at the start, beyond the table's far edge
OFFER_POINT = (0.355, -0.15, TOP + 0.13)  # where the hand holds the block out (the pinch centre)
hand.CONFIG.update(block="block", offer=tuple(o - s for o, s in zip(OFFER_POINT, HAND_START, strict=True)),
                   table_top=TOP)
_GX, _GZ = hand.GRIP_AT
BLOCK_START = (HAND_START[0] - _GX, HAND_START[1], HAND_START[2] - _GZ)  # its origin, pinched by its +x end

OBJECTS = [
    tp.part("rail", "rail", RAIL_ORIGIN),
    hand.part("hand", HAND_START),
    tp.part("block", "block", BLOCK_START),
]

# -- scoring thresholds ----------------------------------------------------------------------------------------
SAFE_CONTACT_N = 10.0  # the robot may never press on the hand harder than this (summed over its links, any moment)
RELEASED_N = 0.1  # success needs the block released: all robot fingertips on it together under this (N)
SAFE_LOAD_N = 5.0  # the net force on the hand from the robot and the block together may never exceed this: what a force
# sensor between the prop hand and its stand reads (the hand's own pinch is internal and does not count). A clean
# handover peaks at about 1.6 N, while the robot grips
RAIL_MAX_SHIFT = 10 * MM  # the rail's centre may move less than this from where it started ...
RAIL_MAX_TURN_DEG = 3.0  # ... and turn less than this
SEAT_RANGE = (-0.5 * MM, 0.8 * MM)  # the base above the lip tops at each lip edge: seated (latched: 0.3 mm)
FEET_IN = 1.0 * MM  # each foot spread at most this far: its tooth is under the lip (it clears the edge at 2 mm)
PADS_SQUEEZED = 0.5 * MM  # each seat pad squeezed at least this far: the lip is clamped between it and the tooth
REST_MAX = 8.0 * MM  # each foot's base at most this far above the lips: the block rests on the rail
LATCH_MAX_ACROSS = 1.0 * MM  # the block's centre from the rail's centreline, latched
REST_MAX_ACROSS = 4.0 * MM
MAX_SQUARE_DEG = 3.0  # latched: square to the rail (yaw) and level (tilt)
REST_MAX_TILT_DEG = 10.0
CREDIT = {"handed_over": 0.25, "resting": 0.5, "latched": 0.75}
CAP = CREDIT["handed_over"]  # dropped, or the rail moved

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


def block_on_rail(objects, name: str = "block") -> dict:
    """The block's pose in the rail's frame: across/along offsets, its base's height above the lips at each lip
    edge, its feet's spread and seat pads' squeeze, its yaw and tilt relative to the rail."""
    p_r, r_r = rail_pose(objects)
    p_b, r_b = _origin(objects[name], C_BLOCK)
    rel, pos = r_r.T @ r_b, r_r.T @ (p_b - p_r)
    feet = [(pos + rel @ np.array((s * tp.LIP_EDGE, 0.0, 0.0)))[2] - tp.RAIL_HEIGHT for s in (1, -1)]
    return {
        "across": float(pos[0]),
        "along": float(pos[1]),
        "feet": [float(f) for f in feet],
        "feet_spread": [float(t) for t in getattr(objects[name], "feet_spread", (math.nan, math.nan))],
        "pads_squeeze": [float(t) for t in getattr(objects[name], "pads_squeeze", (math.nan, math.nan))],
        "yaw_deg": math.degrees(math.atan2(rel[1, 0], rel[0, 0])),
        "tilt_deg": math.degrees(math.acos(min(1.0, max(-1.0, rel[2, 2])))),
    }


def _between_brackets(b) -> bool:
    """Between the right bracket and the beige block (the span a block can be mounted in)."""
    return tp.RIGHT_BRACKET_Y[1] + tp.BLOCK_WIDTH / 2 - 1 * MM <= b["along"] <= tp.BEIGE_FACE_Y - tp.BLOCK_WIDTH / 2 + 1 * MM


def latched(b) -> bool:
    """Both feet clicked under the lips: seated flat on the lips, centred and square, between the brackets, both feet
    back in and both seat pads squeezed by the lips (the clamp)."""
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


def handed_over(h) -> bool:
    """The hand let go of the block into the robot's grip (it lets go only after both robot fingertips have pressed
    the block for 0.3 s) and the block never left the hand before that."""
    return bool(h.released and not h.snatched)


def safe(h) -> bool:
    """Checked after every physics step: the robot never pressed on the hand with more than 10 N, and the load on the
    hand (the net force on it from the robot and the block, including anything pushed or pulled through the block)
    never exceeded 5 N."""
    return h.peak_contact_n <= SAFE_CONTACT_N and h.peak_load_n <= SAFE_LOAD_N


def stage(objects) -> str | None:
    h = objects["hand"]
    if not (safe(h) and handed_over(h)):
        return None
    b = block_on_rail(objects)
    if latched(b):
        return "latched"
    if resting(b):
        return "resting"
    return "handed_over"


def success(objects) -> bool:
    """Handed over safely and never dropped, then latched on the rail, released (no robot fingertip on it) and still,
    held 1 s, the rail in place."""
    h = objects["hand"]
    released = h.touch_n < RELEASED_N
    return bool(stage(objects) == "latched" and released and not h.dropped and rail_ok(objects)
                and objects["block"].resting())


def score(objects) -> float:
    """Partial credit at the end of the episode (success scores 1.0): 0.75 latched, 0.5 resting on the rail, 0.25
    handed over cleanly; 0 if the robot pressed on the hand with more than 10 N, loaded the hand with more than 5 N
    (directly or through the block), or took the block without the hand letting go. Capped at 0.25 if the block was dropped or the rail moved 1 cm or turned 3 degrees."""
    st = stage(objects)
    credit = CREDIT.get(st, 0.0)
    if objects["hand"].dropped or not rail_ok(objects):
        credit = min(credit, CAP)
    return credit


def distance(objects) -> float:
    """Before the handover, metres from the block to the hand's offer point; after it, from the block to the rail's
    nearest mounting point (centred over the rail, between the brackets)."""
    h = objects["hand"]
    if not h.released:
        return float(np.linalg.norm(np.asarray(objects["block"].pos) - np.asarray(OFFER_POINT)))
    b = block_on_rail(objects)
    lo = tp.RIGHT_BRACKET_Y[1] + tp.BLOCK_WIDTH / 2
    hi = tp.BEIGE_FACE_Y - tp.BLOCK_WIDTH / 2
    along = min(max(b["along"], lo), hi)
    return float(math.sqrt(b["across"] ** 2 + (b["along"] - along) ** 2 + max(0.0, max(b["feet"])) ** 2))
