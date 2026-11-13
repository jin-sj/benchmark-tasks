"""On the station table, in front of the arms: a wall board standing across the table's far side, and on it a
wall-mounted split air conditioner's indoor unit (white plastic, 700 mm wide, 270 mm tall, 190 mm deep), its
underside 80 mm above the table. Its front cover is a panel hinged along its top edge, with a finger pull along its
bottom edge; its hinge is stiff, so the cover stays at whatever angle it is left, up to fully open (80 degrees,
swung out and up).

Behind the cover, a mesh air filter covers the unit's coil: 600 mm wide and 168 mm tall, a grey plastic frame with a
mesh, flat and upright along its lower part, its top part bent back 15 degrees. It stands on a ledge behind three
small lugs (6 mm tall), its top tucked behind a lip under the unit's top; two pull tabs stick out of its bottom edge.
This filter is dusty (brown-grey mesh).

On the table in front of the unit lies a clean filter of the same kind (light grey mesh), face up, its bottom edge
(with the pull tabs, sticking up) toward the robot.

Task: replace the dusty filter with the clean one: open the cover, take the dusty filter out, fit the clean filter in
its place, and close the cover.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tb_parts as tp  # noqa: E402
import ac_parts as ap  # noqa: E402

from yam_arena.task_file import TABLE_TOP  # noqa: E402

ap.install()
tp.install()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"

MM = 1e-3
FACE_X = 0.55  # the unit's front, from the arm bases
UNIT_ORIGIN = (FACE_X, 0.0, TOP)
DUSTY_ORIGIN = (FACE_X + ap.SEAT_X, 0.0, TOP + ap.LEDGE)  # fitted
# the clean filter lies face up, its bottom edge toward the arms; its bent top part makes it rest on its bottom edge
# and its top edge, tilted this much from flat
LIE_DEG = math.degrees(math.atan2(ap.FILTER_H, ap.bend()[0]))
CLEAN_X = 0.36  # its bottom edge (the tabs), from the arm bases
CLEAN_ORIGIN = (CLEAN_X, 0.0, TOP + ap.THICK * math.sin(math.radians(LIE_DEG)))

OBJECTS = [ap.unit_part("unit", UNIT_ORIGIN),
           tp.part("filter_dusty", ap.FILTER_DUSTY, DUSTY_ORIGIN),
           tp.part("filter_clean", ap.FILTER_CLEAN, CLEAN_ORIGIN, tilt=math.radians(LIE_DEG))]

# -- scoring --------------------------------------------------------------------------------------------------------
FITTED_POS = 2 * MM  # fitted: the filter within this of its place, up (on the ledge) ...
FITTED_IN = (ap.SEAT_X - ap.LUG[0] + 0.5 * MM, ap.GAP + 0.5 * MM)  # ... between the lugs and the coil (its play) ...
FITTED_Y = 10 * MM  # ... and across ...
FITTED_DEG = 3.0  # ... and turned less than this from its fitted orientation
SHUT_DEG = 3.0  # the cover is shut: within this of closed


def fitted(state) -> bool:
    """The filter in its place: on the ledge, behind the lugs, its top behind the lip, facing out, centred."""
    o, rot = ap.frame(state)
    turned = math.degrees(math.acos(float(np.clip((np.trace(rot) - 1) / 2, -1.0, 1.0))))
    d = o - np.asarray(DUSTY_ORIGIN, dtype=float)
    return bool(turned <= FITTED_DEG and -FITTED_IN[0] <= d[0] <= FITTED_IN[1] and abs(d[2]) <= FITTED_POS
                and abs(d[1]) <= FITTED_Y)


def in_unit(state) -> bool:
    """Any part of the filter behind the unit's front, between its underside and its top (in the filter's place)."""
    p = ap.points(state)
    inside = (p[:, 0] > FACE_X + 0.5 * MM) & (p[:, 2] > TOP + ap.BOTTOM) & (p[:, 2] < TOP + ap.TOP)
    return bool(inside.any())


def cover_shut(objects) -> bool:
    return objects["unit"].cover_deg <= SHUT_DEG


def success(objects) -> bool:
    """The dusty filter out of the unit, the clean filter fitted, the cover shut."""
    return bool(not in_unit(objects["filter_dusty"]) and fitted(objects["filter_clean"]) and cover_shut(objects))


def score(objects) -> float:
    """At the end of the episode (success, held for 1 s, scores 1.0): 0 if any part of the dusty filter is in the
    unit. Otherwise 1.0 the clean filter fitted and the cover shut; 0.5 the clean filter fitted, the cover open;
    0.25 the dusty filter out, the clean one not fitted."""
    if in_unit(objects["filter_dusty"]):
        return 0.0
    if fitted(objects["filter_clean"]):
        return 1.0 if cover_shut(objects) else 0.5
    return 0.25


def distance(objects) -> float:
    """Metres from the clean filter's origin to its fitted place (plus 0.25 m while the dusty filter is in the unit,
    and 0.05 m while the cover is open)."""
    o, _ = ap.frame(objects["filter_clean"])
    return float(np.linalg.norm(o - np.asarray(DUSTY_ORIGIN)) + (0.25 if in_unit(objects["filter_dusty"]) else 0.0)
                 + (0.0 if cover_shut(objects) else 0.05))
