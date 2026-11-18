"""An assembly-line station: the table top is the line pallet of a powertrain sub-assembly, the harness already routed
and clipped. In front of the right arm: a sensor module (cast aluminium, 44 x 52 x 62 mm) fixed on a steel
bracket plate, with a black 3-way sealed connector header on its top, its mouth 83 mm above the table and its lock
ramp facing the robot; a relay box stands beside it on the left and a coolant hose runs behind it. Nearer the robot and to the right, the harness plug (black, 18 x 25 x 30 mm, an orange CPA
slider on the side facing the robot, standing 4 mm proud of the plug's top) stands on its face on the table at the
end of its lead: 190 mm of 7 mm convoluted tube running away from the robot to a harness clip on a post, which also
holds the harness trunk.

The plug goes onto the header one way round (a key rib). Pushed fully on, its latch clicks over the header's lock
ramp; the CPA can then be pushed home (flush with the plug's top), which locks the latch. On a plug that is not fully
on and latched, the CPA will not go home.

Task: connect the harness plug to the sensor and close its orange CPA lock.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tb_parts as tp  # noqa: E402
import cpa_parts as cp  # noqa: E402

from yam_arena.task_file import TABLE_TOP  # noqa: E402

cp.install()
tp.install()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 2.0
SCORE_OVER = "end"

MM = 1e-3
YAW = math.pi  # both parts: the connector's +x (the latch side) toward the robot
HEADER_AT = (0.36, -0.12, TOP)  # the header part's origin: on the table under the shroud's centre
PLUG_AT = (0.26, -0.20, TOP + 0.3 * MM)  # the plug's frame (the middle of its face), standing on the table

OBJECTS = [cp.part("sensor", cp.HEADER, HEADER_AT, YAW), cp.part("plug", cp.PLUG, PLUG_AT, YAW)]

# -- scoring constants ---------------------------------------------------------------------------------------------
SEAT_GAP_MM = 1.5  # latched: the collar within this of the shroud's rim ...
LATCH_REST_MM = 0.5  # ... and the latch arm sprung back to within this of its rest (its hook under the catch face)
ON_AXIS_MM = 1.0  # ... with the plug on the header's axis (its face's middle within this of the axis)
CPA_HOME_MM = 3.0  # the CPA home: pushed down at least this far of its 4.0 mm (within 1 mm of flush)


def _header_frame():
    """(origin of the cavity floor's middle, rotation) of the header's connector frame, in the task's frame."""
    c, s = math.cos(YAW), math.sin(YAW)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.array(HEADER_AT) + np.array([0.0, 0.0, cp.FLOOR * MM]), rot


def plug_in_header(objects) -> tuple[np.ndarray, float]:
    """The plug's face middle in the header's frame (mm: x, y, height above the cavity floor) and its tilt from the
    mating axis (degrees)."""
    o, rot = _header_frame()
    p = objects["plug"]
    d = rot.T @ (np.asarray(p.origin) - o) / MM
    prot = rot.T @ np.asarray(p.rot)
    return d, math.degrees(math.acos(min(1.0, max(-1.0, prot[2, 2]))))


def seat_gap_mm(objects) -> float:
    """The collar's gap above the shroud's rim (mm; 0 when the plug is pushed fully on)."""
    d, _ = plug_in_header(objects)
    return float(d[2] - cp.SEAT)


def latched(objects) -> bool:
    """The plug fully on and its latch caught: on the header's axis (within 1.0 mm), the collar within 1.5 mm of the
    rim, the latch arm sprung back to within 0.5 mm of its rest."""
    d, _ = plug_in_header(objects)
    return bool(math.hypot(d[0], d[1]) <= ON_AXIS_MM and d[2] - cp.SEAT <= SEAT_GAP_MM
                and objects["plug"].latch_mm <= LATCH_REST_MM)


def cpa_home(objects) -> bool:
    return bool(objects["plug"].cpa_mm >= CPA_HOME_MM)


# -- the CPA's pre-lock and detent (once per control step) -----------------------------------------------------------
def _cpa(base, states) -> None:
    if "plug" not in states:
        return
    cp.update_cpa(base.scene.articulations["plug"], latched(states))
    d, _ = plug_in_header(states)
    if math.hypot(d[0], d[1]) <= 5.0:  # over the cavity: the terminals' friction draws the seal plate back with it
        cp.follow_plug(base.scene.articulations["sensor"], float(d[2]))


tp.STEP_HOOKS.append(_cpa)


def success(objects) -> bool:
    """Latched (fully on: collar within 1.5 mm of the rim, latch arm within 0.5 mm of rest) and the CPA home."""
    return latched(objects) and cpa_home(objects)


def score(objects) -> float:
    """At the end: 1.0 the plug latched on the header (on its axis within 1.0 mm, the collar within 1.5 mm of the
    shroud's rim, the latch arm sprung back to within 0.5 mm of rest) and its CPA home (pushed down at least 3.0 of its
    4.0 mm); 0.5 latched, the CPA not home; 0 otherwise (off the header, or on it but not latched: half-mated)."""
    if not latched(objects):
        return 0.0
    return 1.0 if cpa_home(objects) else 0.5


def distance(objects) -> float:
    """How far the plug's face is from its seated position (m), plus the CPA's travel still to go."""
    d, _ = plug_in_header(objects)
    return float(np.linalg.norm(d - np.array([0.0, 0.0, cp.SEAT])) * MM
                 + max(0.0, cp.CPA_TRAVEL - objects["plug"].cpa_mm) * MM)
