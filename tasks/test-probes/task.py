"""On the station table, in front of the arms: a steel TS35 x 7.5 top-hat DIN rail, 50 cm long, screwed down flat
(open side up) and running left to right, as in a control panel. Latched on it at its middle, side by side, a strip
of 40 terminal blocks (5.2 mm apart centre to centre), each 56 mm long across the rail and 40 mm tall above it. From
the left: a green-yellow earth terminal, a grey line (L) terminal, 35 beige terminals of other circuits, a blue
neutral (N) terminal, a green-yellow earth terminal and a beige terminal. On top of each block, at its middle, is a
test opening: a 2.6 mm square hole, 8 mm deep, with a small funnel at its mouth; its floor is the block's metal
clamp. A 12 V DC source is wired into the L (+) and N (-) clamps; no other terminal is wired.

In front of the strip, between the arms, a handheld digital multimeter (autoranging, 6000 counts) lies face up in its
rubber holster, 170 x 90 x 44 mm, its display (22 mm digits) at the end nearer the strip. It is switched off: its
rotary selector (a knob with a grip bar, clicking between OFF, V~, V DC, mV and Ohm, 30 degrees apart, clockwise)
points at OFF. Its two test probes are plugged into it (red into V, black into COM; the leads are not simulated).

In front of the arms, two probe holders (black, 30 mm cubes), one in front of each arm, each holding a test probe
standing tip down: red on the left, black on the right. A probe is 140 mm long: a 2 mm steel tip 15 mm long, a short
shroud, a 16 mm finger guard resting on the holder's top, and a 10 mm handle. Lifting a probe 25 mm frees it from its
holder.

Task: measure the DC voltage between the L terminal and the N terminal with the multimeter: the meter must show it
steadily for at least one second while one probe's tip is held in each terminal's test opening. Do not put a probe
tip into any other terminal's opening.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tb_parts as tp  # noqa: E402
import probe_parts as pp  # noqa: E402
import meter_parts as mp  # noqa: E402

from yam_arena import station  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

tp.RAIL_BEIGE = False
tp.RAIL_LENGTH = 0.50
tp.LEFT_BRACKET_Y = (0.200, 0.208)
tp.RIGHT_BRACKET_Y = (-0.208, -0.200)
pp.install()
mp.install()
tp.install()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 0.0  # the 1 s is counted in the meter's readings (three in a row)
SCORE_OVER = "end"

MM = 1e-3
RAIL_ORIGIN = (0.30, 0.0, TOP + 0.2 * MM)
STRIP_ORIGIN = (RAIL_ORIGIN[0], RAIL_ORIGIN[1], RAIL_ORIGIN[2] + tp.RAIL_HEIGHT)
HOLDERS = {"left": (0.20, 0.14, TOP), "right": (0.20, -0.14, TOP)}
PROBES = {pp.PROBE_RED: "left", pp.PROBE_BLACK: "right"}
METER_ORIGIN = (0.17, 0.0, TOP)  # the meter's underside centre: its display end 2 cm short of the strip
pp.CONFIG.update(strip_origin=STRIP_ORIGIN, bases=station.BASES, meter="meter")

OBJECTS = [tp.part("rail", "rail", RAIL_ORIGIN, fixed=True), tp.part("strip", pp.STRIP, STRIP_ORIGIN)]
OBJECTS += [tp.part(f"holder_{side}", pp.HOLDER, pos) for side, pos in HOLDERS.items()]
OBJECTS += [tp.part(name, name, pp.probe_in_holder(HOLDERS[side])) for name, side in PROBES.items()]
OBJECTS += [mp.part("meter", METER_ORIGIN)]

# -- scoring --------------------------------------------------------------------------------------------------------
CREDIT = {"one_reading": 0.5, "inserted": 0.25}


def record() -> dict:
    return pp.RECORD


def wrong_opening() -> bool:
    """A probe tip entered (1 mm or more below the mouth, inside its bore) the test opening of any terminal other
    than L and N."""
    return pp.RECORD.get("wrong") is not None


def measured() -> bool:
    """The meter, its selector on V DC, showed the test voltage (11.40 to 12.60 V, either sign) at three readings in a
    row (it reads twice a second: at least 1 s), both probes held by a closed gripper at each of them."""
    return pp.RECORD.get("best_streak", 0) >= pp.STEADY_READINGS


def success(objects) -> bool:
    """Measured (the meter showed the test voltage for 1 s, both probes held), no tip ever in a wrong opening."""
    return bool(pp.RECORD and measured() and not wrong_opening())


def score(objects) -> float:
    """At the end of the episode (success scores 1.0): 0 if a probe tip ever entered any other terminal's test
    opening. Otherwise 1.0 measured (on V DC the meter showed the test voltage, 11.40 to 12.60 V either sign, at three
    readings in a row, at least 1 s, both probes held at each); 0.5 it showed the test voltage at one reading at
    least, both probes held; 0.25 a held probe's tip was put into the L or N test opening, 4 mm or more below its
    mouth; 0 otherwise."""
    if not pp.RECORD or wrong_opening():
        return 0.0
    if measured():
        return 1.0
    if pp.RECORD["test_reading_step"] is not None:
        return CREDIT["one_reading"]
    if any(v is not None for v in pp.RECORD["inserted_ever"].values()):
        return CREDIT["inserted"]
    return 0.0


def distance(objects) -> float:
    """Metres from the red probe's tip to the L opening plus from the black probe's tip to the N opening."""
    o = pp._openings_world()
    return float(sum(np.linalg.norm(pp._tip(objects[n]) - o[i])
                     for n, i in zip(pp.CONFIG["probes"], pp.CONFIG["targets"], strict=True)))
