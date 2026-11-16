"""On the station table, in front of the right arm: an open cardboard case (a tray with 40 mm walls) of six soup cans
standing upright, loose, two rows of three (each can 66 mm across, 100 mm tall and 0.42 kg, 70 mm apart; on each lid
a seam ring around a flat centre panel 54 mm across). To the right of it, nearer the robot, a hand price labeler
(blue, 0.2 kg) stands nose down in a dark grey stand, its handle level along the table and its black trigger in front
of the handle.

The labeler: squeezing the trigger fully against the handle and letting it go feeds one price label (white with an
orange band, 25 x 15 mm) out under the roller at its nose; pressing the roller onto a surface then sticks the label
there. Feeding another label while one is already out spoils the first (it is discarded).

Task: price the cans: stick one price label flat on each can's lid, inside the lid's seam ring.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tb_parts as tp  # noqa: E402
import labeler_parts as lp  # noqa: E402

from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_arena.urdf import quat_xyzw_to_matrix  # noqa: E402

lp.install()
tp.install()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 0.0
SCORE_OVER = "end"

MM = 1e-3
CASE_ORIGIN = (0.30, -0.12, TOP)
STAND_ORIGIN = (0.22, -0.30, TOP)
LABELER_ORIGIN = lp.labeler_in_stand(STAND_ORIGIN)
SHELF_ORIGIN = (0.40, 0.0, TOP - 0.25)  # the labeler's roll: out of sight under the table
LABEL_NAMES = [f"label_{i}" for i in range(lp.N_LABELS)]
CLEAR = 5 * MM  # a stuck label shows once the roller is this far above it ...
CLEAR_SIDE = 25 * MM  # ... or this far to its side


def _shelf_slot(i: int) -> tuple[float, float, float]:
    return (SHELF_ORIGIN[0], SHELF_ORIGIN[1] - 0.145 + i * 0.0155, SHELF_ORIGIN[2] + 0.005)


CAN_NAMES = [f"can_{i + 1}" for i in range(lp.ROWS * lp.COLS)]
CAN_STARTS = [(CASE_ORIGIN[0] + x, CASE_ORIGIN[1] + y) for x, y in lp.can_centres()]
WASTE_SPOT = (STAND_ORIGIN[0] - 0.07, STAND_ORIGIN[1] + 0.02)  # spoiled labels are discarded here, by the stand

OBJECTS = [tp.part("case", lp.CASE, CASE_ORIGIN), tp.part("stand", lp.STAND, STAND_ORIGIN),
           tp.part("shelf", lp.SHELF, SHELF_ORIGIN), lp.labeler_part("labeler", LABELER_ORIGIN)]
OBJECTS += [tp.part(n, lp.CAN, (x, y, TOP + lp.CASE_BASE)) for n, (x, y) in zip(CAN_NAMES, CAN_STARTS, strict=True)]
OBJECTS += [tp.part(n, lp.LABEL, _shelf_slot(i)) for i, n in enumerate(LABEL_NAMES)]

LID_TOP = TOP + lp.LID_Z  # the lids' tops, the cans standing in the case


# -- the labeler's feed, and the labels' sticking (once per control step) --------------------------------------------
RECORD: dict = {}


def _new_record() -> dict:
    return {"next": 0, "out": False, "armed": True, "fed": 0, "stuck": [], "pending": None, "glued": {},
            "spoiled": [], "step": -1}


def _pose(state) -> tuple[np.ndarray, np.ndarray]:
    """(origin, rotation) of a part from its ``ObjectState`` (whose pos is its bounds' centre)."""
    rot = quat_xyzw_to_matrix(np.asarray(state.quat_xyzw, dtype=float))
    return np.asarray(state.pos, dtype=float), rot


def _can_frame(state) -> tuple[np.ndarray, np.ndarray]:
    """(origin: the middle of the can's bottom, rotation)."""
    c, rot = _pose(state)
    return c - rot @ np.array(tp.local_center(lp.CAN)), rot


def _nose(state) -> tuple[np.ndarray, np.ndarray]:
    """The nose roller's lowest point (world) and the labeler's rotation."""
    rot = quat_xyzw_to_matrix(np.asarray(state.quat_xyzw, dtype=float))
    return np.asarray(state.origin, dtype=float) + rot @ lp.NOSE, rot


def _surface_below(p, states) -> tuple[float, str | None]:
    """(height, can name or None) of the highest surface under point ``p``: a standing can's lid, the case's floor,
    or the table."""
    for n in CAN_NAMES:
        o, rot = _can_frame(states[n])
        if rot[2, 2] >= math.cos(math.radians(UPRIGHT_DEG)) and math.hypot(p[0] - o[0], p[1] - o[1]) <= lp.CAN_R:
            return float(o[2] + lp.CAN_H), n
    hx, hy = lp.CASE_HALF
    if abs(p[0] - CASE_ORIGIN[0]) <= hx and abs(p[1] - CASE_ORIGIN[1]) <= hy:
        return TOP + lp.CASE_BASE, None
    return TOP, None


def _place(base, name: str, p, yaw: float) -> None:
    """Put label ``name`` flat at ``p`` (its underside's middle), its long side along ``yaw``."""
    import torch

    obj = base.scene.rigid_objects[name]
    o = base.scene.env_origins[0]
    c = np.array(tp.local_center(lp.LABEL))
    rc = np.array([c[0] * math.cos(yaw) - c[1] * math.sin(yaw), c[0] * math.sin(yaw) + c[1] * math.cos(yaw), c[2]])
    pose = torch.tensor([[p[0] + rc[0], p[1] + rc[1], p[2] + rc[2] + 0.05 * MM, 0.0, 0.0, math.sin(yaw / 2),
                          math.cos(yaw / 2)]], dtype=torch.float32, device=o.device)
    pose[0, :3] += o
    obj.write_root_pose_to_sim_index(root_pose=pose)
    obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=o.device))


def _glued_pose(states, can: str, local) -> tuple[np.ndarray, float]:
    """A glued label's place in the world: (its underside's middle, its yaw), from its can's pose."""
    o, rot = _can_frame(states[can])
    x, y, yaw = local
    p = o + rot @ np.array([x, y, lp.CAN_H])
    return p, yaw + math.atan2(rot[1, 0], rot[0, 0])


def _feed(base, states) -> None:
    step = int(base.episode_length_buf[0])
    if not RECORD or step <= 1 or step < RECORD.get("step", 0):
        RECORD.clear()
        RECORD.update(_new_record())
    if step == RECORD["step"] or "labeler" not in states:
        return
    RECORD["step"] = step
    lab = states["labeler"]
    nose, rot = _nose(lab)
    yaw = math.atan2(rot[1, 1], rot[0, 1])  # the handle's direction: the label's long side
    if lab.trigger_mm >= lp.FIRE_MM and RECORD["armed"]:
        RECORD["armed"] = False
    elif lab.trigger_mm <= lp.REARM_MM and not RECORD["armed"]:  # released after a full squeeze: a label comes out
        RECORD["armed"] = True
        if RECORD["out"]:  # the one already out is spoiled: discarded
            i = RECORD["next"] - 1
            _place(base, LABEL_NAMES[i], (WASTE_SPOT[0], WASTE_SPOT[1] + 0.02 * len(RECORD["spoiled"]), TOP), 0.0)
            RECORD["spoiled"].append(i)
            RECORD["out"] = False
        if RECORD["next"] < len(LABEL_NAMES):
            RECORD["next"] += 1
            RECORD["out"] = True
            RECORD["fed"] += 1
    if RECORD["out"] and rot[2, 2] >= math.cos(math.radians(lp.UPRIGHT_DEG)):
        top, can = _surface_below(nose, states)
        if nose[2] - top <= lp.STICK_GAP:  # the roller pressed on a surface: the label sticks there
            RECORD["pending"] = (RECORD["next"] - 1, (nose[0], nose[1], top), yaw, top, can)
            RECORD["stuck"].append((RECORD["next"] - 1, step, can or "elsewhere"))
            RECORD["out"] = False
    if RECORD["pending"] is not None:  # shown once the roller has lifted off it (the label stays where it was pressed)
        i, at, at_yaw, top, can = RECORD["pending"]
        if nose[2] - top > CLEAR or math.hypot(nose[0] - at[0], nose[1] - at[1]) > CLEAR_SIDE:
            if can is None:
                _place(base, LABEL_NAMES[i], at, at_yaw)
            else:  # on a can's lid: glued to the can, in its frame
                o, crot = _can_frame(states[can])
                d = crot.T @ (np.asarray(at) - o)
                RECORD["glued"][i] = (can, (float(d[0]), float(d[1]), at_yaw - math.atan2(crot[1, 0], crot[0, 0])))
            RECORD["pending"] = None
    for i, (can, local) in RECORD["glued"].items():  # glued labels go where their cans go
        p, y = _glued_pose(states, can, local)
        _place(base, LABEL_NAMES[i], p, y)


tp.STEP_HOOKS.append(_feed)


# -- scoring (the end state) -------------------------------------------------------------------------------------------
UPRIGHT_DEG = 10.0  # a can standing up: its axis within this of vertical


def _can_ok(states, can: str) -> bool:
    """The can stands upright in its case."""
    o, rot = _can_frame(states[can])
    hx, hy = lp.CASE_HALF
    return bool(rot[2, 2] >= math.cos(math.radians(UPRIGHT_DEG)) and abs(o[0] - CASE_ORIGIN[0]) <= hx
                and abs(o[1] - CASE_ORIGIN[1]) <= hy and o[2] < TOP + 0.03)


def labels_on_cans(objects) -> tuple[list[list[bool]], int]:
    """For each can, its labels (True: wholly inside the lid's seam ring); and how many labels are elsewhere (stray:
    stuck anywhere else, or spoiled)."""
    per_can = {n: [] for n in CAN_NAMES}
    glued = RECORD.get("glued", {})
    for i, (can, (x, y, yaw)) in glued.items():
        lx, ly, _ = lp.LABEL_SIZE
        c, s_ = math.cos(yaw), math.sin(yaw)
        corners = [(x + c * a - s_ * b, y + s_ * a + c * b) for a in (-lx / 2, lx / 2) for b in (-ly / 2, ly / 2)]
        per_can[can].append(all(math.hypot(u, v) <= lp.PANEL_R for u, v in corners))
    used = len(RECORD.get("stuck", [])) + len(RECORD.get("spoiled", []))
    stray = used - len(glued)
    return [per_can[n] for n in CAN_NAMES], stray


def priced(objects) -> int:
    """How many cans stand upright in the case with exactly one label, wholly inside the lid's seam ring."""
    per_can, _ = labels_on_cans(objects)
    return sum(1 for n, labels in zip(CAN_NAMES, per_can, strict=True) if labels == [True] and _can_ok(objects, n))


def success(objects) -> bool:
    """All six cans priced (upright in the case, one label each, inside the seam ring), no label anywhere else."""
    _, stray = labels_on_cans(objects)
    return bool(stray == 0 and priced(objects) == len(CAN_NAMES))


def score(objects) -> float:
    """At the end: 1.0 all six cans priced (standing upright in the case, exactly one label each, wholly inside the
    lid's seam ring) and no label anywhere else (none stuck elsewhere or spoiled); 0.5 at least three cans priced; 0.25
    at least one; 0 otherwise."""
    if success(objects):
        return 1.0
    n = priced(objects)
    return 0.5 if n >= 3 else (0.25 if n >= 1 else 0.0)


def distance(objects) -> float:
    """Cans still to price, in units of 0.1 (plus the labeler's nose's distance from the first unpriced can's lid)."""
    per_can, _ = labels_on_cans(objects)
    todo = [n for n, labels in zip(CAN_NAMES, per_can, strict=True) if labels != [True]]
    if not todo:
        return 0.0
    nose, _ = _nose(objects["labeler"])
    o, _ = _can_frame(objects[todo[0]])
    return 0.1 * len(todo) + float(np.linalg.norm(nose - (o + np.array([0.0, 0.0, lp.CAN_H]))))
