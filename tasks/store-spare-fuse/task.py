"""On the station table, in front of the arms: a steel TS35 x 7.5 top-hat DIN rail, 50 cm long, screwed down flat
(open side up) and running left to right, with an end bracket 4 cm in from each end. At the rail's middle, latched
side by side, nine terminal blocks; from the left: two blue and one grey plain block, three grey fuse terminal
blocks, two beige and one grey plain block.

A fuse block is 8.2 mm thick (along the rail), 62 mm long (across it) and 34 mm tall above the rail; the plain
blocks are thinner (5.1 or 6.2 mm), shorter (52 mm) and lower (28 mm). Each fuse block carries an upright fuse-holder
plate (25 mm long, 24 mm tall) on the far half of its top and, in the near half of its top, 14.5 mm toward the arms
from the rail's centreline, an empty spare-fuse pocket: a square hole 6 mm wide and 14 mm deep.

In front of the right arm, nearer the arms than the rail: a black rack, 30 x 30 x 12 mm, with one spare 5 x 20 mm
glass cartridge fuse standing upright in it, 12 mm of the fuse out of the rack.

Task: put the spare fuse into the spare-fuse pocket of the middle one of the three fuse blocks. Stored, the fuse
stands on the pocket's floor, 6 mm of it out of the block's top.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tb_parts as tp  # noqa: E402
import fuse_strip as fs  # noqa: E402

from yam_arena.task_file import TABLE_TOP  # noqa: E402

tp.RAIL_BEIGE = False
tp.RAIL_LENGTH = 0.50
tp.LEFT_BRACKET_Y = (0.200, 0.208)
tp.RIGHT_BRACKET_Y = (-0.208, -0.200)
fs.install()
tp.install()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"

MM = 1e-3
RAIL_ORIGIN = (0.30, -0.04, TOP + 0.2 * MM)
STRIP_ORIGIN = (RAIL_ORIGIN[0], RAIL_ORIGIN[1], RAIL_ORIGIN[2] + tp.RAIL_HEIGHT)
RACK_ORIGIN = (0.18, -0.22, TOP)  # nearer the arms than the rail, in front of the right arm
FUSE_START = tuple(float(a + b) for a, b in zip(RACK_ORIGIN, fs.FUSE_IN_RACK, strict=True))
TARGET = 1  # the middle one of the three fuse blocks (0 = the leftmost fuse block)

OBJECTS = [tp.part("rail", "rail", RAIL_ORIGIN, fixed=True), tp.part("strip", fs.STRIP, STRIP_ORIGIN),
           tp.part("rack", fs.RACK, RACK_ORIGIN), tp.part("fuse", "fuse", FUSE_START, tilt=math.pi / 2)]

# -- scoring thresholds ----------------------------------------------------------------------------------------
ON_AXIS = 1.5 * MM  # the fuse's centre within this of the pocket's axis
ENTERED = 2 * MM  # "partly in": the fuse's lower end at least this far below the pocket's mouth
SEATED = 1 * MM  # "stored": the fuse's lower end within this of the pocket's floor
PICKED = 30 * MM  # "picked": the fuse's centre this far from where it started
CREDIT = {"stored": 1.0, "partly": 0.5, "picked": 0.25}


def pocket_world(k: int = TARGET) -> tuple[np.ndarray, float, float]:
    """A fuse block's pocket in the world: its axis (x, y), its mouth's and its floor's height."""
    x, y, mouth, floor = fs.pocket(k)
    return np.array([STRIP_ORIGIN[0] + x, STRIP_ORIGIN[1] + y]), STRIP_ORIGIN[2] + mouth, STRIP_ORIGIN[2] + floor


def fuse_moved(objects) -> float:
    return float(np.linalg.norm(np.asarray(objects["fuse"].pos, dtype=float) - np.asarray(FUSE_START)))


def in_pocket(objects, k: int = TARGET) -> float | None:
    """How far (m) the fuse's lower end is above the pocket's floor, if the fuse is in pocket k (its centre on the
    pocket's axis and its lower end below the mouth); else None."""
    axis, mouth, floor = pocket_world(k)
    st = objects["fuse"]
    c = np.asarray(st.pos, dtype=float)
    low = float(np.asarray(st.low)[2])
    if float(np.linalg.norm(c[:2] - axis)) <= ON_AXIS and low <= mouth - ENTERED:
        return low - floor
    return None


def stage(objects) -> str | None:
    gap = in_pocket(objects)
    if gap is not None:
        return "stored" if gap <= SEATED else "partly"
    return "picked" if fuse_moved(objects) >= PICKED else None


def success(objects) -> bool:
    """The fuse standing on the floor of the middle fuse block's pocket, and still (held 1 s)."""
    return bool(stage(objects) == "stored" and objects["fuse"].resting())


def score(objects) -> float:
    """At the end of the episode (success scores 1.0): 1.0 the fuse stored in the middle fuse block's pocket (its
    lower end within 1 mm of the pocket's floor); 0.5 partly in that pocket (its lower end 2 mm or more below the
    mouth); 0.25 taken from the rack (its centre 30 mm or more from where it started), wherever it is, including in
    another block's pocket; 0 otherwise."""
    return CREDIT.get(stage(objects), 0.0)


def distance(objects) -> float:
    """Metres from the fuse's centre to the pocket's axis."""
    return float(np.linalg.norm(np.asarray(objects["fuse"].pos, dtype=float)[:2] - pocket_world()[0]))
