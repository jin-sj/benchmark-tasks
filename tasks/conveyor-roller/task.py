"""On the station table, in front of the arms: an installed section of a light-duty gravity roller conveyor running away
from the robot, its frame bolted down on the table: two blue steel angle rails 381 mm apart (15 in between frames),
63.5 mm tall, 610 mm long, each with a row of hex holes 44.5 mm above the table, every 114.3 mm. Five galvanized steel
rollers (48.3 mm across, 368 mm long) sit in the holes on their hex axles, their tops 68.6 mm above the table. The
second roller from the robot is worn: dull and rust-stained, a flat scored band round its middle and a red paint mark
near its left end; its neighbours are 114.3 mm away on either side.

Each roller's 7/16 in hex axle (11.1 mm across flats, 419 mm long) sticks out 25.4 mm past each end of the tube and
through the rails' hex holes (11.9 mm across flats, flats on top and bottom). The axle is spring-retained: it slides up
to 16 mm through the roller toward the roller's left end against a spring (4 N to start, 13.6 N at full travel), and
springs back. The tube spins freely on its axle.

To the right of the conveyor, on the table: a blue plastic parts tray (466 x 86 mm, 25 mm walls) nearer the conveyor,
holding a new roller (bright galvanized, the same size) lying along the table's length, its left end (the end the
axle slides toward) away from the robot; beyond it, a grey reject tray of the same size, empty.

Task: replace the worn roller with the new one, and put the worn roller in the reject tray.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tb_parts as tp  # noqa: E402
import roller_parts as rp  # noqa: E402

from yam_arena.task_file import TABLE_TOP  # noqa: E402

rp.install()
tp.install()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"

MM = 1e-3
SLOT_X = 0.30  # the worn roller's position (its axle's centre)
CONVEYOR_ORIGIN = (SLOT_X - rp.roller_xs()[rp.WORN_INDEX], 0.0, TOP)
HOLE = (SLOT_X, TOP + rp.HOLE_Z)  # the empty slot's holes' centre (x, z)
SEATED_DROP = 0.4 * MM  # a seated hex axle rests on the hole's bottom flat, this far below the hole's centre
TRAY_ORIGIN = (0.33, -0.30, TOP)
REJECT_ORIGIN = (0.33, -0.42, TOP)
NEW_YAW = -math.pi / 2  # the new roller lies along x, its left end (+y in its frame) away from the robot
rp.INIT_TWIST["new_roller"] = math.radians(rp.HEX_TWIST_NEW)

OBJECTS = [
    tp.part("conveyor", rp.CONVEYOR, CONVEYOR_ORIGIN),
    tp.part("parts_tray", rp.TRAY, TRAY_ORIGIN),
    tp.part("reject_tray", rp.REJECT, REJECT_ORIGIN),
    rp.roller_part("worn_roller", rp.WORN, (SLOT_X, 0.0, TOP + rp.HOLE_Z - SEATED_DROP)),
    rp.roller_part("new_roller", rp.ROLLER, (TRAY_ORIGIN[0], TRAY_ORIGIN[1], TOP + rp.TRAY_FLOOR + rp.TUBE_R + 0.5 * MM),
                   yaw=NEW_YAW),
]

# -- scoring -----------------------------------------------------------------------------------------------------------
THROUGH = 5 * MM  # an axle end counts as seated when it sticks out this far past the rail's outer face ...
IN_HOLE = 3 * MM  # ... within this of the hole's centre (x and z)


def ends_seated(objects, name: str = "new_roller") -> int:
    """How many of the roller's axle ends are through the empty slot's hex holes: the axle crosses the rail's web
    within 3 mm of the hole's centre and its end sticks out at least 5 mm past the rail's outer face (either end in
    either rail: the roller works either way round)."""
    st = objects[name]
    a, b = np.asarray(st.tip_left, dtype=float), np.asarray(st.tip_right, dtype=float)
    n = 0
    for tip, other in ((a, b), (b, a)):
        side = 1.0 if tip[1] > 0 else -1.0
        mid = side * (rp.INNER_Y + rp.WEB_T / 2)  # the web's mid-plane
        if abs(tip[1]) < rp.OUTER_Y + THROUGH or abs(tip[1] - other[1]) < 1e-6:
            continue
        p = tip + (other - tip) * (tip[1] - mid) / (tip[1] - other[1])  # where the axle crosses the web
        if abs(p[0] - HOLE[0]) <= IN_HOLE and abs(p[2] - HOLE[1]) <= IN_HOLE:
            n += 1
    return n


RESTING = TOP + rp.TRAY_FLOOR + rp.TUBE_R + 5 * MM  # a roller lying on the tray's floor has its middle below this


def in_reject_tray(objects, name: str = "worn_roller") -> bool:
    """The roller lies in the reject tray: its tube's middle inside the tray's walls, the tube resting on the tray's
    floor (its middle less than 5 mm above where it is when lying on the floor)."""
    p = np.asarray(objects[name].pos, dtype=float)
    ix, iy = rp.TRAY_INNER
    return bool(abs(p[0] - REJECT_ORIGIN[0]) <= ix / 2 and abs(p[1] - REJECT_ORIGIN[1]) <= iy / 2 and p[2] <= RESTING)


def success(objects) -> bool:
    """The worn roller in the reject tray, and the new roller's axle out through both holes of its slot."""
    return in_reject_tray(objects) and ends_seated(objects) == 2


def score(objects) -> float:
    """At the end: 1.0 the worn roller lies in the reject tray and both of the new roller's axle ends stick out
    through the slot's hex holes at least 5 mm past the rails' outer faces; 0.5 the worn roller in the tray and one
    end of the new roller's axle through its hole so; 0.25 the worn roller in the tray; 0 otherwise."""
    if not in_reject_tray(objects):
        return 0.0
    return {2: 1.0, 1: 0.5}.get(ends_seated(objects), 0.25)


def distance(objects) -> float:
    """Metres still to go: the worn roller's middle from the reject tray, plus the new roller's middle from the slot."""
    w = np.asarray(objects["worn_roller"].pos, dtype=float)
    nr = np.asarray(objects["new_roller"].pos, dtype=float)
    d_w = 0.0 if in_reject_tray(objects) else float(np.linalg.norm(w[:2] - np.array(REJECT_ORIGIN[:2])))
    d_n = float(np.linalg.norm(nr - np.array([SLOT_X, 0.0, TOP + rp.HOLE_Z])))
    return d_w + d_n
