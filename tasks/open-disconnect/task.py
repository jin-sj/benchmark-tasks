"""On the station table, in front of the arms: a steel TS35 x 7.5 top-hat DIN rail, 50 cm long, screwed down flat
(open side up) and running left to right, with an end bracket 4 cm in from each end. At the rail's middle, side by
side, five identical grey knife-disconnect terminal blocks, latched on the rail.

A block is 6.2 mm thick (along the rail), 62 mm long (across it) and 34 mm tall above the rail. On its top is an
orange lever, 4 mm wide and 18 mm long, hinged near the block's middle. Closed, the lever lies flat along the block's
top, its free end toward the arms, with a 6 mm upright tab at that end. Pushed up and away from the arms it swings
about its hinge (an axis along the rail) and, past 45 degrees, snaps open: it then stands upright (95 degrees) and
the block's circuit is interrupted. Moving it takes about 1 to 2 N. All five levers start closed.

Task: open the disconnect lever of the second block from the left, and leave the other four closed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tb_parts as tp  # noqa: E402
import disconnect as dc  # noqa: E402

from yam_arena.task_file import TABLE_TOP  # noqa: E402

tp.RAIL_BEIGE = False
tp.RAIL_LENGTH = 0.50
tp.LEFT_BRACKET_Y = (0.200, 0.208)
tp.RIGHT_BRACKET_Y = (-0.208, -0.200)
dc.install()
tp.install()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"

MM = 1e-3
RAIL_ORIGIN = (0.30, -0.04, TOP + 0.2 * MM)
STRIP_ORIGIN = (RAIL_ORIGIN[0], RAIL_ORIGIN[1], RAIL_ORIGIN[2] + tp.RAIL_HEIGHT)
TARGET = 1  # the second block from the left (0 = leftmost)

OBJECTS = [tp.part("rail", "rail", RAIL_ORIGIN, fixed=True), dc.part("strip", STRIP_ORIGIN)]

# -- scoring thresholds ----------------------------------------------------------------------------------------
OPEN_DEG = 80.0  # a lever at or beyond this is open (its stop is at 95)
TOUCHED_DEG = 10.0  # another lever lifted beyond this at any point in the episode: its circuit was interrupted
SNAP_DEG = 45.0  # the target lever lifted past this at any point: it was opened (it snaps open past here)
LIFTED_DEG = 15.0  # the target lever lifted at least this at any point: progress
HOLD_S = 1.0  # full credit needs the target open, the others never touched, for this long
CREDIT = {"opened": 0.5, "lifted": 0.25}


def levers(objects) -> tuple[float, ...]:
    return tuple(objects["strip"].levers_deg)


def others_touched(objects) -> bool:
    """Whether any of the other four levers was lifted more than 10 degrees at any point so far (its circuit was
    interrupted, even if it was closed again)."""
    return any(a > TOUCHED_DEG for i, a in enumerate(objects["strip"].levers_max_deg) if i != TARGET)


def target_open(objects) -> bool:
    return levers(objects)[TARGET] >= OPEN_DEG


def success(objects) -> bool:
    """The second lever open, the other four never lifted more than 10 degrees (held 1 s)."""
    return bool(target_open(objects) and not others_touched(objects))


def score(objects) -> float:
    """At the end of the episode (success scores 1.0); 0 if any of the other four levers was lifted more than 10
    degrees at any point. Otherwise: 1.0 the second lever open (80 degrees or more) and kept open for the last 1 s;
    0.5 it was lifted past its snap point (45 degrees) at some point; 0.25 it was lifted at least 15 degrees at some
    point; 0 otherwise."""
    if others_touched(objects):
        return 0.0
    st = objects["strip"]
    if target_open(objects) and st.levers_open_s[TARGET] >= HOLD_S:
        return 1.0
    reached = st.levers_max_deg[TARGET]
    if reached >= SNAP_DEG:
        return CREDIT["opened"]
    return CREDIT["lifted"] if reached >= LIFTED_DEG else 0.0


def distance(objects) -> float:
    """Degrees the second lever is short of open."""
    return max(0.0, OPEN_DEG - levers(objects)[TARGET])
