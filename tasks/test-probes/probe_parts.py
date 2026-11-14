"""A panel's terminal strip with test openings, two multimeter test probes and their holders, and the per-step record
the scorer reads.

- Strip (one fixed part, ``STRIP``): 40 terminal blocks of this task's own generic design, latched side by side on a
  fixed rail, 5.2 mm apart centre to centre. Each block's top has a test opening at its middle: a square hole
  ``OPENING`` wide with a 45 degree lead-in, ``OPENING_DEPTH`` deep, its floor the block's metal clamp. Colours from
  the left: a green-yellow earth block, the grey L block, then beige blocks, and near the right end the blue N block
  and a green-yellow earth block.
- Probes (free parts, ``PROBE_RED`` and ``PROBE_BLACK``): a steel tip, a tapered shroud, a finger guard and a handle,
  140 mm long. Frame: origin at the tip's end, the probe's axis along +z.
- Holders (fixed parts, ``HOLDER``): a block with a hole the shroud fits into; a probe stands in it tip down, resting
  on its finger guard.

The record (``RECORD``), updated once per control step from the probes' poses and the robot's joint readings: each
probe's tip position, which opening (if any) it is in and how deep, whether a gripper is closed on its handle, the
largest lift out of its holder so far, whether a tip has entered a wrong opening so far, and how long both tips have
been seated and held without a break.
"""

from __future__ import annotations

import math

import numpy as np

import tb_parts as tp

MM = 1e-3
STRIP, HOLDER, PROBE_RED, PROBE_BLACK = "test_strip", "probe_holder", "probe_red", "probe_black"

# -- the strip (strip frame: origin on the rail's lip-top plane at the strip's middle; x across the rail, away from
# the arms; y along it, the robot's left; z up) -------------------------------------------------------------------
N_BLOCKS = 40
PITCH = 5.2 * MM
SEAM = 0.2 * MM  # gap drawn between neighbouring housings
HOUSING = (56 * MM, 40 * MM)  # length across the rail, height above the lip tops
COLOURS = ["green_yellow", "grey"] + ["beige"] * 35 + ["blue", "green_yellow", "beige"]
L_INDEX, N_INDEX = 1, 37  # from the left (0 = leftmost): 187 mm apart (closer than about 17 cm, the wrists touch)
OPENING = 2.6 * MM  # the test opening's square bore
OPENING_DEPTH = 8.0 * MM  # from the block's top to the clamp at its floor
LEAD_IN = 1.5 * MM  # 45 degree lead-in (a funnel) at its mouth, on its near and far edges ...
LEAD_IN_SIDE = 1.0 * MM  # ... and on its side edges (the walls along the rail are 1.2 mm thick)


def block_y(i: int) -> float:
    return ((N_BLOCKS - 1) / 2 - i) * PITCH


def opening(i: int) -> tuple[float, float, float]:
    """Block i's test opening in the strip frame: (x, y) of its axis and the height of its mouth."""
    return 0.0, block_y(i), HOUSING[1]


def strip_shapes() -> list[tp.Shape]:
    out = []
    h0, h1 = -OPENING / 2, OPENING / 2
    top, lead, floor = HOUSING[1], HOUSING[1] - LEAD_IN, HOUSING[1] - OPENING_DEPTH
    for i, look in enumerate(COLOURS):
        y = block_y(i)
        s0, s1 = y - PITCH / 2 + SEAM / 2, y + PITCH / 2 - SEAM / 2
        b0, b1 = y - OPENING / 2, y + OPENING / 2
        x0, x1 = -HOUSING[0] / 2, HOUSING[0] / 2
        box = lambda *v, look=look: tp._box(STRIP, look, "polyamide", *v)  # noqa: E731
        out += [
            box(x0, x1, s0, s1, 0.3 * MM, floor),  # below the opening (its top is the clamp)
            box(x0, h0 - LEAD_IN, s0, s1, floor, top), box(h1 + LEAD_IN, x1, s0, s1, floor, top),  # near and far
            box(h0 - LEAD_IN, h0, s0, s1, floor, lead), box(h1, h1 + LEAD_IN, s0, s1, floor, lead),
            box(h0, h1, s0, b0, floor, top - LEAD_IN_SIDE), box(h0, h1, b1, s1, floor, top - LEAD_IN_SIDE),  # sides
            box(h0, h1, s0, b0 - LEAD_IN_SIDE, top - LEAD_IN_SIDE, top),
            box(h0, h1, b1 + LEAD_IN_SIDE, s1, top - LEAD_IN_SIDE, top),
            tp._prism(STRIP, look, "polyamide", [(h0 - LEAD_IN, top), (h0 - LEAD_IN, lead), (h0, lead)], b0, b1),
            tp._prism(STRIP, look, "polyamide", [(h1, lead), (h1 + LEAD_IN, lead), (h1 + LEAD_IN, top)], b0, b1),
            tp.Shape(STRIP, look, "polyamide", profile=((b0 - LEAD_IN_SIDE, top), (b0 - LEAD_IN_SIDE, top - LEAD_IN_SIDE),
                                                         (b0, top - LEAD_IN_SIDE)), span=(h0, h1), axis="X"),
            tp.Shape(STRIP, look, "polyamide", profile=((b1, top - LEAD_IN_SIDE), (b1 + LEAD_IN_SIDE, top - LEAD_IN_SIDE),
                                                         (b1 + LEAD_IN_SIDE, top)), span=(h0, h1), axis="X"),
            tp._box(STRIP, "steel", None, h0, h1, b0, b1, floor, floor + 0.2 * MM),  # the clamp's metal (look)
            # looks: wire entries on both ends' faces
            tp._box(STRIP, "hole", None, x0 - 0.05 * MM, x0 + 0.1 * MM, y - 1.5 * MM, y + 1.5 * MM, 18 * MM, 24 * MM),
            tp._box(STRIP, "hole", None, x1 - 0.1 * MM, x1 + 0.05 * MM, y - 1.5 * MM, y + 1.5 * MM, 18 * MM, 24 * MM),
        ]
    return out


# -- the probes (probe frame: origin at the tip's end, axis +z) -----------------------------------------------------
TIP = (2.0 * MM, 15 * MM)  # square tip: width, length
SHROUD = (6.0 * MM, 10 * MM)  # above the tip: width, length
GUARD = (16 * MM, 3 * MM)  # finger guard: width, thickness
HANDLE = (10 * MM, 100 * MM)  # width, length
CAP = (6.0 * MM, 12 * MM)  # strain relief at the top
PROBE_MASS = 0.03
Z_SHROUD = TIP[1]
Z_GUARD = Z_SHROUD + SHROUD[1]
Z_HANDLE = Z_GUARD + GUARD[1]
Z_CAP = Z_HANDLE + HANDLE[1]
PROBE_LENGTH = Z_CAP + CAP[1]


def _probe_shapes(colour: str) -> list[tp.Shape]:
    sq = lambda w, z0, z1, look, mat: tp._box("probe", look, mat, -w / 2, w / 2, -w / 2, w / 2, z0, z1)  # noqa: E731
    return [
        sq(TIP[0], 0.0, TIP[1], "steel", "steel"),
        sq(SHROUD[0], Z_SHROUD, Z_GUARD, colour, "plastic"),
        sq(GUARD[0], Z_GUARD, Z_HANDLE, colour, "plastic"),
        sq(HANDLE[0], Z_HANDLE, Z_CAP, colour, "rubber"),
        sq(CAP[0], Z_CAP, PROBE_LENGTH, colour, "plastic"),
    ]


def red_shapes():
    return [tp.Shape(PROBE_RED, s.look, s.material, s.center, s.size) for s in _probe_shapes("red")]


def black_shapes():
    return [tp.Shape(PROBE_BLACK, s.look, s.material, s.center, s.size) for s in _probe_shapes("black")]


# -- the holder (holder frame: origin at the middle of its underside) -----------------------------------------------
HOLDER_SIZE = (30 * MM, 30 * MM, 30 * MM)
HOLDER_HOLE = 8.0 * MM  # the shroud (6 mm) stands in it; the finger guard rests on the holder's top


def holder_shapes() -> list[tp.Shape]:
    hx, hy, h = HOLDER_SIZE[0] / 2, HOLDER_SIZE[1] / 2, HOLDER_SIZE[2]
    b = HOLDER_HOLE / 2
    f = h - Z_GUARD + 1 * MM  # the hole's floor: the tip hangs 1 mm above it
    box = lambda *v: tp._box(HOLDER, "tray", "plastic", *v)  # noqa: E731
    return [box(-hx, hx, -hy, hy, 0.0, f), box(-hx, -b, -hy, hy, f, h), box(b, hx, -hy, hy, f, h),
            box(-b, b, -hy, -b, f, h), box(-b, b, b, hy, f, h)]


def probe_in_holder(holder_origin) -> tuple[float, float, float]:
    """Where a probe's origin (its tip's end) sits when standing in a holder at ``holder_origin``."""
    x, y, z = holder_origin
    return float(x), float(y), float(z + HOLDER_SIZE[2] - Z_GUARD)


def install() -> None:
    tp.SHAPES[STRIP] = strip_shapes
    tp.SHAPES[HOLDER] = holder_shapes
    tp.SHAPES[PROBE_RED] = red_shapes
    tp.SHAPES[PROBE_BLACK] = black_shapes
    tp.FIXED_KINDS.update((STRIP, HOLDER))
    tp.FREE_KINDS.update({PROBE_RED: PROBE_MASS, PROBE_BLACK: PROBE_MASS})
    tp.LOOKS.update({"red": ((0.80, 0.10, 0.08), 0.5, 0.0), "black": ((0.06, 0.06, 0.07), 0.5, 0.0),
                     "green_yellow": ((0.35, 0.65, 0.20), 0.5, 0.0)})
    tp.STEP_HOOKS.append(_record)


# -- the record -------------------------------------------------------------------------------------------------------
ON_CLAMP = 0.7 * MM  # a tip in its opening's bore within this of the opening's floor, the clamp, touches the clamp
SEATED_DEPTH = OPENING_DEPTH - ON_CLAMP  # (that is, this far below the mouth): it makes contact
ENTERED_DEPTH = 1.0 * MM  # a tip this far below a wrong opening's mouth, inside its bore, has entered it
IN_BORE = OPENING / 2 + 0.3 * MM  # a tip's centre within this of an opening's axis (each way) is inside its bore
LIFTED = 30 * MM  # a probe whose tip rose this far above where it started has been lifted out of its holder
HELD_AXIS = 4 * MM  # held: a closed gripper's TCP within this of the probe's axis ...
HELD_GAP = HANDLE[0] + 1.5 * MM  # ... its pads closed to at most this ...
HELD_SPAN = (Z_HANDLE - 5 * MM, Z_CAP + 5 * MM)  # ... and the TCP along the handle

CONFIG = {"strip_origin": None, "probes": (PROBE_RED, PROBE_BLACK), "targets": (L_INDEX, N_INDEX),
          "robots": ("left_robot", "right_robot"), "bases": None}
RECORD: dict = {}


def _new_record(states) -> dict:
    tips = {n: _tip(states[n]) for n in CONFIG["probes"]}
    return {"start_tip": tips, "lift": {n: 0.0 for n in tips}, "wrong": None, "both_s": 0.0, "now": {},
            "seated_ever": {n: None for n in tips}}


def _pose(state):
    from yam_arena.urdf import quat_xyzw_to_matrix

    rot = quat_xyzw_to_matrix(np.asarray(state.quat_xyzw, dtype=float))
    lo, hi = tp.bounds(PROBE_RED)
    centre = np.array([(a + b) / 2 for a, b in zip(lo, hi, strict=True)])
    return np.asarray(state.pos, dtype=float) - rot @ centre, rot  # the probe's origin (tip end), rotation


def _tip(state) -> np.ndarray:
    return _pose(state)[0]


def _openings_world() -> list[np.ndarray]:
    so = np.asarray(CONFIG["strip_origin"], dtype=float)
    return [so + np.array(opening(i)) for i in range(N_BLOCKS)]


def which_opening(tip) -> tuple[int | None, float]:
    """(index of the opening whose bore the tip is in, how far below its mouth); (None, 0) if in none."""
    for i, o in enumerate(_openings_world()):
        if abs(tip[0] - o[0]) <= IN_BORE and abs(tip[1] - o[1]) <= IN_BORE and tip[2] < o[2]:
            return i, float(o[2] - tip[2])
    return None, 0.0


def _tcps(base) -> list[tuple[np.ndarray, float]]:
    """Each robot arm's TCP position and gripper gap (m), from its joint readings."""
    from yam_arena import ik
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    out = []
    for robot, side in zip(CONFIG["robots"], ("left", "right"), strict=True):
        art = base.scene.articulations[robot]
        q = art.data.joint_pos.torch[0].detach().cpu().numpy().astype(np.float64)
        names = list(art.joint_names)
        arm = np.array([q[names.index(j)] for j in ARM_JOINTS])
        gap = sum(abs(q[names.index(j)]) for j in FINGER_JOINTS)
        out.append((ik.tcp_pose(arm, CONFIG["bases"][side])[0], gap))
    return out


def held(state, tcps) -> bool:
    """A gripper closed on the probe's handle: its TCP within 4 mm of the probe's axis, along the handle, its pads
    closed to at most the handle's width plus 1.5 mm."""
    o, rot = _pose(state)
    axis = rot[:, 2]
    for p, gap in tcps:
        d = p - o
        along = float(d @ axis)
        off = float(np.linalg.norm(d - along * axis))
        if gap <= HELD_GAP and off <= HELD_AXIS and HELD_SPAN[0] <= along <= HELD_SPAN[1]:
            return True
    return False


def _record(base, states) -> None:
    step = int(base.episode_length_buf[0])
    if not RECORD or step <= 1 or step < RECORD.get("step", 0):
        RECORD.clear()
        RECORD.update(_new_record(states))
    RECORD["step"] = step
    tcps = _tcps(base)
    targets = dict(zip(CONFIG["probes"], CONFIG["targets"], strict=True))
    now = {}
    for n in CONFIG["probes"]:
        tip = _tip(states[n])
        i, depth = which_opening(tip)
        h = held(states[n], tcps)
        RECORD["lift"][n] = max(RECORD["lift"][n], float(tip[2] - RECORD["start_tip"][n][2]))
        if i is not None and i not in CONFIG["targets"] and depth >= ENTERED_DEPTH and RECORD["wrong"] is None:
            RECORD["wrong"] = (n, i, step)
        seated = i is not None and i in CONFIG["targets"] and depth >= SEATED_DEPTH
        if seated and h and RECORD["seated_ever"][n] is None:
            RECORD["seated_ever"][n] = (i, step)
        now[n] = {"tip": [round(float(v), 4) for v in tip], "opening": i, "depth_mm": round(depth / MM, 2),
                  "held": h, "seated": seated}
    ok = all(now[n]["seated"] and now[n]["held"] for n in CONFIG["probes"]) and \
        {now[n]["opening"] for n in CONFIG["probes"]} == set(CONFIG["targets"])
    RECORD["both_s"] = RECORD["both_s"] + float(base.step_dt) if ok else 0.0
    RECORD["now"] = now
