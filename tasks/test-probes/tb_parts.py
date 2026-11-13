"""The task's own parts: a TS35 DIN rail assembly, generic terminal blocks with sprung feet, and a parts tray.

yam_arena task files support single primitives and library assets only, and yam_arena has no extension point
for multi-shape or jointed parts. This module is the task's own minimal mechanism, kept in the task folder:

- :class:`Part` is a :class:`yam_arena.task_file.Box` (so the stock task-file loader accepts it and the
  scorer gets an ``ObjectState`` for it) that also names which part it is.
- :func:`install` wraps ``yam_arena.scene.task_object`` and ``object_states`` the moment that module is imported
  (it imports Isaac Sim, so it is never imported here before the simulator starts). Every :class:`Part` is then
  spawned from a USD file this module writes; every other object goes through yam_arena unchanged.

The parts (dimensions in mm in the comments, metres in code):

- rail (one rigid body, free or fixed): a steel TS35 x 7.5 top-hat rail lying web-down, 1 mm sheet but 2 mm
  lips (thickened so a pressed tooth cannot tunnel through them at 240 Hz), lip outer edges at +-17.5 from the
  centreline. Fixed to it: an end bracket near each end and a beige terminal block
  mounted against the left bracket.
- block (an articulation of five bodies: housing, two feet, two seat pads): a generic terminal block of this
  project's own design, in two models. "SYM": a screw-clamp feed-through block with a sprung foot at each end,
  5.1 thick, 60 long across the rail, 46.25 tall from the bottom of its feet (39.5 above the rail). "HOOK": a
  push-in block with a fixed hook at one end and a sprung foot at the other, 5.1 thick, 65 long, 36 tall. Their
  outlines are simple polygons coded below as plain coordinates. Each sprung foot is its own link on a prismatic
  joint across the rail, sprung inward by a joint drive: a stand-in for a real foot's plastic flexure beams.
- tray (a fixed collider): an open tray with low walls and no slots.

The latch (a spring hack, not a flexure model), all flat-on-flat contacts, which PhysX resolves exactly:
- the click: each foot is its own link on a joint across the rail, sprung inward (1 N/mm). Pressing the block
  down, the lip edges ride up the feet's lead-ins and spread them; once the tooth tips pass under the lips, the
  springs snap the feet back in.
- the clamp: under the housing, over each lip, a seat pad on a sprung vertical joint (3 N/mm) stands 1.2 mm
  proud when free. Pressing the block on squeezes the pads flush, which lets the teeth pass 0.3 mm below the lips'
  undersides; let go, the pads push the block back up until the teeth bear on the lips from below, 2.7 N per
  side (feet and pads slide on the lips with friction 0.25: a latched block slides at ~2.2 N). A latched block
  therefore sits 0.3 mm proud of the lips without play, resists sliding along the rail
  (friction on the pads and under the lips), does not rock, and stays where it is left.
A latched block cannot be pulled straight off (the tooth tops are flat, as on the real terminal): a foot has to be
spread first. Each foot's arm stands 0.3 mm outside its lip edge, which keeps the block centred.

Frames: the rail's origin is the centre of its web's underside (x across the rail, y along it, z up); a
block's origin is on the rail's centreline at the plane of the lip tops when seated (x across the rail, y
through its 5.1 mm thickness, z up); the tray's origin is the centre of its underside.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import inspect
import math
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from yam_arena import task_file

MM = 1e-3

# -- rail (TS35 x 7.5, EN 60715) ----------------------------------------------------------------------------
RAIL_LENGTH = 240 * MM
RAIL_HEIGHT = 7.5 * MM  # top of the lips above the table
SHEET = 1.0 * MM  # web and side walls
LIP_THICK = 2.0 * MM  # simulation: a real rail's lips are 1 mm; at 240 Hz a 1 mm lip tunnels through a pressed
# tooth instead of spreading the foot
LIP_EDGE = 17.5 * MM  # lips' outer edges, from the centreline
LIP_INNER = 13.5 * MM  # outer faces of the side walls, where the lips start
RAIL_MASS = 0.15  # kg: 24 cm of 1 mm steel rail (~0.1) plus two end brackets and the mounted block
# Along the rail (+y is the left end, seen from behind the arms).
LEFT_BRACKET_Y = (69.0 * MM, 77.0 * MM)  # the free stub beyond it (77..120) is where the left arm holds the rail
RIGHT_BRACKET_Y = (-118.0 * MM, -110.0 * MM)
BRACKET_HALF_X = 22.0 * MM  # at the base; 12 mm at the top
BRACKET_TOP = 45.0 * MM
HOLD_Y = 105.0 * MM  # where the left arm presses the rail (in the stub, 36 mm from the mounted block)

# -- terminal block: generic screw-clamp feed-through outline (this project's own design) ---------------------------
# Section coordinates are (across, up) in mm in the design frame: across 0..60 over the block's length, up from the
# bottom of its feet. The rail's centreline is at across 32 (the block sits 2 mm off-centre on the rail) and the lip
# tops at up 6.76 when seated. Block frame: x = (across - 32) mm, z = (up - 6.76) mm.
_ACROSS0, _UP0 = 32.0, 6.76
# Housing outline (visual), counter-clockwise: lower wings over the feet, the wire-entry steps, upper wings, two screw
# shafts between posts, and a cross-connection channel in the middle.
_HOUSING_OUTLINE = (
    (17.0, 6.76), (47.0, 6.76), (47.0, 12.0), (58.0, 12.0), (58.0, 20.0), (47.0, 22.4), (47.0, 26.3), (55.0, 33.0),
    (55.0, 37.7), (49.0, 37.7), (49.0, 46.25), (44.0, 46.25), (44.0, 38.7), (41.0, 38.7), (41.0, 46.25), (39.0, 46.25),
    (39.0, 37.7), (36.0, 34.7), (36.0, 29.0), (28.0, 29.0), (28.0, 34.7), (25.0, 37.7), (25.0, 46.25), (23.0, 46.25),
    (23.0, 38.7), (20.0, 38.7), (20.0, 46.25), (15.0, 46.25), (15.0, 37.7), (9.0, 37.7), (9.0, 33.0), (17.0, 26.3),
    (17.0, 22.4), (6.0, 20.0), (6.0, 12.0), (17.0, 12.0),
)
# Housing collision: convex pieces covering the outline, each at least 1 mm thick so PhysX does not inflate it.
_HOUSING_HULLS = (
    ((17.0, 6.76), (47.0, 6.76), (47.0, 26.3), (17.0, 26.3)),  # body, up to the wire entries
    ((6.0, 12.0), (17.0, 12.0), (17.0, 22.4), (6.0, 20.0)),  # left lower wing
    ((47.0, 12.0), (58.0, 12.0), (58.0, 20.0), (47.0, 22.4)),  # right lower wing
    ((9.0, 33.0), (17.0, 26.3), (17.0, 37.7), (9.0, 37.7)),  # left upper wing
    ((17.0, 26.3), (28.0, 26.3), (28.0, 34.7), (25.0, 37.7), (17.0, 37.7)),  # left shoulder
    ((15.0, 37.7), (20.0, 37.7), (20.0, 46.25), (15.0, 46.25)),  # left outer post
    ((20.0, 37.7), (25.0, 37.7), (25.0, 38.7), (20.0, 38.7)),  # left screw shaft floor
    ((23.0, 38.7), (25.0, 38.7), (25.0, 46.25), (23.0, 46.25)),  # left inner post
    ((28.0, 26.3), (36.0, 26.3), (36.0, 29.0), (28.0, 29.0)),  # floor of the cross-connection channel
    ((47.0, 26.3), (55.0, 33.0), (55.0, 37.7), (47.0, 37.7)),  # right upper wing
    ((36.0, 26.3), (47.0, 26.3), (47.0, 37.7), (39.0, 37.7), (36.0, 34.7)),  # right shoulder
    ((44.0, 37.7), (49.0, 37.7), (49.0, 46.25), (44.0, 46.25)),  # right outer post
    ((39.0, 37.7), (44.0, 37.7), (44.0, 38.7), (39.0, 38.7)),  # right screw shaft floor
    ((39.0, 38.7), (41.0, 38.7), (41.0, 46.25), (39.0, 46.25)),  # right inner post
)
_SCREWS = (22.5, 41.5)  # across: clamping screws in the shafts, heads on the shaft floor (up 38.7)
_WIRE_ENTRIES = ((17.0, -1), (47.0, 1))  # across, facing: the vertical steps between the wings (up 22.4..26.3)
# Openings through the section (looks only): a cross-connection hole and two test holes.
_OPENINGS = (
    ((30.5, 13.5), (33.5, 13.5), (33.5, 16.5), (30.5, 16.5)),
    ((8.5, 15.0), (10.0, 15.0), (10.0, 16.5), (8.5, 16.5)),
    ((54.0, 15.0), (55.5, 15.0), (55.5, 16.5), (54.0, 16.5)),
)
BLOCK_WIDTH = 5.1 * MM  # along the rail
BLOCK_LENGTH = 60.0 * MM  # across the rail (x -32 .. +28 in the block frame)
BLOCK_HEIGHT = (46.25 - _UP0) * MM  # above the lip tops: 39.5
BLOCK_MASS = 0.0066  # housing; with both feet 7.6 g, a typical mass for a block this size
FOOT_MASS = 0.0005

# -- terminal block: feet (the latch hack) --------------------------------------------------------------------
# One foot's section as (u, z) in mm, relaxed: u outward from the rail centreline, z up from the lip tops; the same
# for both feet but for the release lever outside the hook (the -x foot has a long one).
FOOT_SPRING = 1000.0  # N/m (1 N/mm) per foot, across the rail: the click
PAD_SPRING = 3000.0  # N/m (3 N/mm) per seat pad, vertical: the clamp
# A 0.5 g foot on these springs rings at ~250 Hz, above the 240 Hz physics rate, and the solver then lets parts
# sink into the lip. Joint armature (inertia along the joint only, no weight) brings it to ~25 Hz.
FOOT_ARMATURE = 0.05  # kg
# The seat pads need it too: without it (or with 2 g of real mass instead) a pad sinks into the lip and the clamp
# barely holds (0.24 N to slide a latched block, against 1.3 N). The price: while a block sits clamped PhysX reports
# a steady velocity for it that it does not have (~20 mm/s and 1.5 rad/s on CPU, 6 mm/s and 0.4 rad/s on GPU, the
# pose unchanged to a micrometre), so _block_state reports the block's motion over the last control step instead.
PAD_ARMATURE = 0.05  # kg
FOOT_DAMPING = 8.0  # N s/m
PAD_DAMPING = 8.0  # N s/m
# The latch's features are made larger than the real part's (hook 1.2 mm, no clamp travel) so that PhysX's
# sub-millimetre contact errors on these small parts stay small against them.
TOOTH_PLAY = 0.3  # mm: with the pads squeezed flush, the tooth tops sit this far below the lips' undersides
TOOTH_REACH = 2.0  # mm: how far the tooth tip reaches inside the lip edge
TIP_CHAMFER = 0.3  # mm: small 45 degree chamfer from the tooth's top down to its tip
TIP_FLAT = 0.6  # mm: vertical face at the tip, between the chamfer and the lead-in
ARM_PLAY = 0.3  # mm: a foot's arm stands this far outside the lip edge, which keeps the block centred
FOOT_TRAVEL = 3.5 * MM  # across; the tooth clears the lip edge at 2.0 mm
# Snap-off variant (None: both feet sprung across the rail, as above). A fixed hook on the +x side and, on the -x
# side (the release-lever foot), a foot on a slide inclined this many degrees below horizontal, outward:
# pulling that end of the block up makes the lip push the tooth down, which drives the foot out along the slide
# against its spring until the tooth slips off the lip. A real screw-clamp block needs a screwdriver in the lever here.
SNAP_OFF_DEG: float | None = None
# Screwdriver variant (FIXED_HOOK and RELEASE_SLOT): the real one-sided latch. The +x foot is a fixed hook; the -x foot
# is the usual sprung slider, with a release slot at the block's near end: a screwdriver blade put in the slot and
# levered (handle tipped toward the block) pushes the foot's tab outward until its tooth comes out from under the lip.
FIXED_HOOK = False
HOOK_FLEX = 0.0  # m: with FIXED_HOOK, how far the hook may flex outward (0: rigid). The plastic gives under a firm push
HOOK_SPRING = 2000.0  # N/m: the hook's flex stiffness (the sprung foot's is FOOT_SPRING)
RELEASE_SLOT = False
SLOT_CHEEK = (31.6 * MM, 30.0 * MM)  # the housing's flat cheek facing the slot: from x -31.6 to -30.0 mm ...
SLOT_Z = (0.5 * MM, 9.0 * MM)  # ... and from z 0.5 to 9 mm
SLOT_WIDTH = 1.6 * MM  # the slot between the cheek and the foot's tab (a 0.7 mm blade goes in)
TAB_THICK = 1.6 * MM  # the foot's tab, outward of the slot
SLOT_FLOOR = 0.0  # the slot's floor (the foot's tab turns in under the slot), its top at this height
SNAP_SPRING = 1000.0  # N/m along the slide
PAD_PROUD = 1.2 * MM  # a free pad's face below the housing's seat; latched it is squeezed to TOOTH_PLAY
PAD_X = (13.6 * MM, 15.4 * MM)  # pads' extent across the rail (over the lip tops, 13.5..17.5)
PAD_THICK = 1.2 * MM
PAD_MASS = 0.0002
SEATED_HEIGHT = TOOTH_PLAY * MM  # a latched block's base above the lip tops
FOOT_HALF_WIDTH = 2.4 * MM  # along the rail: 0.15 mm inside each face, so feet never rub a neighbour
_LIP_U, _LIP_Z = LIP_EDGE / MM, -LIP_THICK / MM


def _foot_profile(side_lever: bool, reach: float | None = None, bottom: float = -6.5):
    """The foot's collision pieces, (u, z) mm: the hook (lead-in, tip, chamfer and flat tooth top), the arm above
    it, and the outer foot with its release lever."""
    top_z = _LIP_Z - TOOTH_PLAY
    tip = _LIP_U - (TOOTH_REACH if reach is None else reach)
    wall, out, top = _LIP_U + ARM_PLAY, 19.2, 0.9
    hook = ((tip + TIP_CHAMFER, top_z), (out, top_z), (out, bottom), (18.5, bottom),
            (tip, top_z - TIP_CHAMFER - TIP_FLAT), (tip, top_z - TIP_CHAMFER))
    arm = ((wall, top_z), (out, top_z), (out, top), (wall, top))
    if side_lever:
        outer = ((out, bottom), (28.5, -5.2), (30.5, -3.5), (30.5, -1.5), (27.0, 0.9), (out, top))
    else:
        outer = ((out, bottom), (26.5, -5.0), (27.5, -2.5), (25.0, 0.9), (out, top))
    return {"hook": hook, "arm": arm, "outer": outer}


def _lead_in_height() -> float:
    """Height (m) of a block's base above the lip tops when it rests unpressed on its relaxed feet's lead-ins."""
    hook = _foot_profile(True)["hook"]
    (bx, bz), (tx, tz) = hook[3], hook[4]
    # the lead-in runs from below the tooth tip (tx, tz) down and out to (bx, bz); it meets the lip's top corner
    z_at_edge = tz + (_LIP_U - tx) * (bz - tz) / (bx - tx)
    return -z_at_edge * MM


RESTING_BASE_HEIGHT = _lead_in_height()
FOOT_BOTTOM = -6.5 * MM  # lowest point of the feet below the base

# Mounted (beige) block: its free face, which the new block must sit flush against.
BEIGE_Y = (LEFT_BRACKET_Y[0] - BLOCK_WIDTH, LEFT_BRACKET_Y[0])
BEIGE_FACE_Y = BEIGE_Y[0]
SLOT_Y = BEIGE_FACE_Y - BLOCK_WIDTH / 2  # centre of a block flush against the beige block

# -- tray ---------------------------------------------------------------------------------------------------
TRAY_INNER = (170.0 * MM, 64.0 * MM)  # x, y
TRAY_WALL = 3.0 * MM
TRAY_FLOOR = 3.0 * MM
TRAY_WALL_HEIGHT = 12.0 * MM  # above the floor

# -- materials and looks ------------------------------------------------------------------------------------
# (static friction, dynamic friction, combine mode). PhysX uses the combine mode of higher priority
# (average < min < multiply < max), so the blocks' "min" governs every contact they make.
MATERIALS = {
    "steel": (0.30, 0.30, "average"),
    "polyamide": (0.35, 0.35, "min"),  # polyamide on steel
    # the feet's and seat pads' contacts with the lips: plastic on steel, a little low. Lower (0.15) a block set on
    # the rail unpressed slides off its lead-ins sideways (they slope outward); higher (0.35) the feet jam and
    # never click. Bench (CPU): clicks at 5.4 N, slides at 2.2 N latched
    "latch": (0.25, 0.25, "min"),
    "plastic": (0.40, 0.40, "average"),
    "rubber": (0.90, 0.90, "average"),  # the screwdriver's handle
    "glass": (0.40, 0.40, "average"),  # the fuse (glass tube, metal caps)
}
LOOKS = {  # sRGB, roughness, metallic
    "steel": ((0.72, 0.74, 0.76), 0.35, 0.8),
    "zinc": ((0.55, 0.57, 0.60), 0.4, 0.6),
    "screw": ((0.78, 0.76, 0.70), 0.3, 0.9),
    "blue": ((0.12, 0.33, 0.78), 0.5, 0.0),
    "beige": ((0.86, 0.80, 0.66), 0.5, 0.0),
    "grey": ((0.55, 0.57, 0.60), 0.5, 0.0),
    "hole": ((0.03, 0.03, 0.035), 0.9, 0.0),
    "tray": ((0.18, 0.19, 0.21), 0.7, 0.0),
    "handle": ((0.80, 0.16, 0.12), 0.6, 0.0),
    "glass": ((0.85, 0.90, 0.92), 0.15, 0.0),
}
CONTACT_OFFSET = 0.5 * MM


# -- shapes -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Shape:
    """One piece of a part, in its body's frame: a box (``center``, ``size``), a prism (an (x, z) ``profile``
    extruded along y over ``span``; convex unless visual-only), or a visual cylinder (``center``, ``radius``,
    ``length`` along ``axis``)."""

    body: str
    look: str
    material: str | None  # None: visual only
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: tuple[float, float, float] | None = None
    profile: tuple[tuple[float, float], ...] | None = None
    span: tuple[float, float] = (0.0, 0.0)
    radius: float = 0.0
    length: float = 0.0
    axis: str = "Z"
    visible: bool = True  # False: a collider hidden behind a separate look

    def points(self) -> list[tuple[float, float, float]]:
        """Corner points (for bounds)."""
        if self.profile is not None:
            if self.axis == "X":  # a prism along x: its profile is in (y, z)
                return [(x, y, z) for y, z in self.profile for x in self.span]
            return [(x, y, z) for x, z in self.profile for y in self.span]
        if self.size is not None:
            return [tuple(c + s * h / 2 for c, h, s in zip(self.center, self.size, sg, strict=True))
                    for sg in ((-1, -1, -1), (1, 1, 1))]
        r, h = self.radius, self.length / 2
        ext = {"X": (h, r, r), "Y": (r, h, r), "Z": (r, r, h)}[self.axis]
        return [tuple(c + s * e for c, e in zip(self.center, ext, strict=True)) for s in (-1, 1)]


def _box(body, look, material, x0, x1, y0, y1, z0, z1) -> Shape:
    return Shape(body, look, material, ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0))


def _prism(body, look, material, profile, y0, y1, visible=True) -> Shape:
    return Shape(body, look, material, profile=tuple((float(x), float(z)) for x, z in profile), span=(y0, y1),
                 visible=visible)


# Generic push-in terminal block with a fixed hook and a sprung foot (this project's own design), in mm, design frame
# (across 0..65.6 from the sprung foot's end, up 0..36.15 from the feet). Rail: lip tops at up 7.05, lip edges at
# across 11.1 and 46.1 (centre 28.6). The left foot is the sprung one, the right a fixed hook.
_HOOK_RAIL = (28.6, 7.05)  # (across, up) of the rail's centreline at the lip tops
_HOOK_BODY_LOOK = (
    ((13.4, 7.05), (42.8, 7.05), (42.8, 12.4), (64.6, 12.4), (65.6, 13.4), (65.6, 25.5), (61.5, 25.5), (61.5, 29.0),
     (57.0, 29.0), (57.0, 36.15), (45.0, 36.15), (45.0, 31.0), (41.0, 31.0), (41.0, 36.15), (34.5, 36.15),
     (33.5, 34.0), (23.5, 34.0), (22.5, 36.15), (16.0, 36.15), (16.0, 31.0), (12.0, 31.0), (12.0, 36.15), (5.0, 36.15),
     (5.0, 29.0), (1.6, 29.0), (1.6, 25.5), (0.8, 25.5), (0.8, 13.4), (1.8, 12.4), (13.4, 12.4)),
)
_HOOK_FOOT_LOOK = (  # the sprung foot and its flexure, which move with the foot
    ((0.8, 0.0), (10.0, 0.0), (12.8, 4.0), (12.8, 5.4), (10.6, 5.4), (10.6, 12.4), (5.2, 12.4), (5.2, 5.4),
     (0.8, 5.4)),
)
_HOOK_HULLS = (  # collision: the housing as two boxes (upper body; the lower middle over the rail)
    ((65.6, 12.4), (65.6, 36.15), (0.8, 36.15), (0.8, 12.4)),
    ((42.8, 7.35), (42.8, 12.4), (13.4, 12.4), (13.4, 7.35)),
)
_HOOK_HOLES = (  # looks: the two wire entries and a test hole
    ((2.6, 17.0), (5.0, 17.0), (5.0, 19.5), (2.6, 19.5)),
    ((61.4, 17.0), (63.8, 17.0), (63.8, 19.5), (61.4, 19.5)),
    ((30.5, 14.0), (33.0, 14.0), (33.0, 16.5), (30.5, 16.5)),
)


BLOCK_MODEL = "SYM"  # "SYM": screw-clamp block (symmetric sprung feet); "HOOK": push-in block (fixed hook + sprung foot, release pocket)
# HOOK block's sprung foot, below the flexure (u outward from the rail centreline, z up from the lip tops, mm): the
# sole, and a release pocket at its outer end (2 mm wide, its floor 2 mm above the foot's underside) between the
# sole's inner part and an upright rim
FOOT_SOLE = ((19.2, 25.0, -7.0, -2.8), (25.0, 27.0, -7.0, -5.0), (27.0, 28.5, -7.0, -1.8))  # u0, u1, z0, z1
HOOK_REACH = 2.5  # mm: the fixed hook's tooth reaches this far in under its lip


def _hook_frame(pts):
    """HOOK design frame (across, up) mm to block-frame (x, z) m."""
    ax, up = _HOOK_RAIL
    return [((a - ax) * MM, (u - up) * MM) for a, u in pts]


def hook_shapes(body: str, look: str, y: float, collide: bool = True, feet: str | None = "own") -> list[Shape]:
    """A HOOK-model push-in terminal block centred at y along the rail, seated (origin on the lip-top plane): its
    outline as the look, two boxes as the housing's collision; feet: a fixed hook at +x and, at -x, the sprung
    foot with the release pocket at its outer end."""
    w = BLOCK_WIDTH / 2
    y0, y1 = y - w, y + w
    mat = "polyamide" if collide else None
    own = feet == "own"
    out = [_prism(body, look, None, _hook_frame(p), y0, y1) for p in _HOOK_BODY_LOOK]
    out += [_prism("foot_n" if own else body, look, None, _hook_frame(p), y0, y1) for p in _HOOK_FOOT_LOOK]
    out += [_prism(body, "hole", None, _hook_frame(p), y0 - 0.05 * MM, y1 + 0.05 * MM) for p in _HOOK_HOLES]
    if collide:
        out += [_prism(body, look, mat, _hook_frame(h), y0, y1, visible=False) for h in _HOOK_HULLS]
    if feet is None:
        return out
    fw0, fw1 = y - FOOT_HALF_WIDTH, y + FOOT_HALF_WIDTH
    fmat = "latch" if own and collide else None
    if own and collide and SNAP_OFF_DEG is not None:  # the snap-off variant's centring rib (see sym_shapes); the HOOK model has none
        x1 = LIP_INNER - SHEET - ARM_PLAY * MM
        out.append(Shape(body, look, mat, (x1 - 0.4 * MM, y, -0.65 * MM), (0.8 * MM, 2 * FOOT_HALF_WIDTH, 1.7 * MM),
                         visible=False))
    for name, s, reach in (("foot_p", 1, HOOK_REACH), ("foot_n", -1, None)):
        fbody = name if own else body
        pieces = _foot_profile(False, reach)
        for key in ("hook", "arm"):
            pts = [(s * u * MM, z * MM) for u, z in pieces[key]]
            out.append(_prism(fbody, look, fmat, pts if s > 0 else pts[::-1], fw0, fw1, visible=False))
        if s < 0:
            for u0, u1, z0, z1 in FOOT_SOLE:
                out.append(Shape(fbody, look, fmat, (-(u0 + u1) / 2 * MM, y, (z0 + z1) / 2 * MM),
                                 ((u1 - u0) * MM, 2 * FOOT_HALF_WIDTH, (z1 - z0) * MM), visible=False))
        if own:  # the seat pad over this lip
            x0, x1 = sorted((s * PAD_X[0], s * PAD_X[1]))
            out.append(_box(name.replace("foot", "pad"), look, fmat, x0, x1, fw0, fw1, -PAD_PROUD, -PAD_PROUD + PAD_THICK))
            out[-1] = Shape(out[-1].body, out[-1].look, out[-1].material, out[-1].center, out[-1].size, visible=False)
    return out


def _drawing(pts, flip=False):
    """SYM design frame (across, up) mm to block-frame (x, z) m; ``flip`` mirrors across the rail centreline."""
    out = [(((a - _ACROSS0) * (-1 if flip else 1)) * MM, (u - _UP0) * MM) for a, u in pts]
    return out[::-1] if flip else out  # keep counter-clockwise


def sym_shapes(body: str, look: str, y: float, collide: bool = True, feet: str | None = "own",
               ) -> list[Shape]:
    """A SYM-model (screw-clamp) terminal block centred at ``y`` along the rail, seated (origin on the lip-top plane).

    ``feet``: "own" puts each foot on its own body (``foot_p``/``foot_n``, for the jointed block); "visual" adds
    them to ``body`` as looks only, with no seat pads (for a block that is part of the rail).
    """
    w = BLOCK_WIDTH / 2
    y0, y1 = y - w, y + w
    mat = "polyamide" if collide else None
    out = [_prism(body, look, None, _drawing(_HOUSING_OUTLINE), y0, y1)]
    out += [_prism(body, look, mat, _drawing(h), y0, y1, visible=False) for h in _HOUSING_HULLS] if collide else []
    # looks: screw heads in the shafts, dark wire entries and through-holes
    for a in _SCREWS:
        x = (a - _ACROSS0) * MM
        out.append(Shape(body, "screw", None, (x, y, (38.7 - _UP0 + 1.0) * MM), radius=1.9 * MM, length=2.0 * MM))
        out.append(_box(body, "hole", None, x - 1.9 * MM, x + 1.9 * MM, y - 0.25 * MM, y + 0.25 * MM,
                        (38.7 - _UP0 + 1.9) * MM, (38.7 - _UP0 + 2.05) * MM))
    for a, facing in _WIRE_ENTRIES:
        x = (a - _ACROSS0) * MM
        out.append(_box(body, "hole", None, *sorted((x, x + facing * 0.06 * MM)), y - 1.8 * MM, y + 1.8 * MM,
                        (22.4 - _UP0 + 0.3) * MM, (26.3 - _UP0 - 0.3) * MM))
    for opening in _OPENINGS:  # dark prisms a little proud of both faces
        out.append(_prism(body, "hole", None, _drawing(opening), y0 - 0.05 * MM, y1 + 0.05 * MM))
    if feet is None:
        return out
    fw0, fw1 = y - FOOT_HALF_WIDTH, y + FOOT_HALF_WIDTH
    if (SNAP_OFF_DEG is not None or FIXED_HOOK) and collide and feet == "own":
        # snap-off variant: a centring rib under the housing, 1.5 mm into the rail's channel 0.3 mm from the inside of
        # the +x side wall (shallow, so it leaves the channel once the block is tipped ~20 degrees): the block cannot shift toward the fixed hook, so a pull on the other end drives the sprung
        # foot out instead of sliding the block
        x1 = LIP_INNER - SHEET - ARM_PLAY * MM
        out.append(_box(body, look, mat, x1 - 0.8 * MM, x1, fw0, fw1, -3.0 * MM, 0.2 * MM).__class__(
            body, look, mat, (x1 - 0.4 * MM, y, -0.65 * MM), (0.8 * MM, 2 * FOOT_HALF_WIDTH, 1.7 * MM), visible=False))
    for name, s, lever in (("foot_p", 1, False), ("foot_n", -1, True)):
        fbody = name if feet == "own" else body
        fmat = "latch" if feet == "own" and collide else None
        # (the fixed hook's tooth keeps the usual 2 mm reach: the centring rib below stops the block shifting toward
        # it, and a short tooth comes out from under its lip with a short move at the end of the dismount)
        reach = None
        # snap-off variant: the sprung foot is 1.5 mm shorter, so sliding out and down along its slide it does not
        # reach the table (the rail lies on it; a full-length foot stops 1.2 mm above it, just short of letting go)
        bottom = -5.0 if (SNAP_OFF_DEG is not None and s < 0) else -6.5
        for piece in _foot_profile(lever, reach, bottom).values():
            pts = [(s * u * MM, z * MM) for u, z in piece]
            out.append(_prism(fbody, look, fmat, pts if s > 0 else pts[::-1], fw0, fw1))
        if feet == "own":  # the seat pad over this lip, free (standing PAD_PROUD below the seat)
            x0, x1 = sorted((s * PAD_X[0], s * PAD_X[1]))
            out.append(_box(name.replace("foot", "pad"), look, fmat, x0, x1, fw0, fw1, -PAD_PROUD,
                            -PAD_PROUD + PAD_THICK))
    if RELEASE_SLOT and feet == "own":
        # the release slot at the near (-x) end: a flat cheek on the housing, and on the sprung foot an L-shaped tab
        # (an upright beyond the slot and a floor under it)
        c0, c1 = SLOT_CHEEK
        t0 = c0 + SLOT_WIDTH
        bmat = "polyamide" if collide else None
        fmat = "latch" if collide else None
        out.append(_box(body, look, bmat, -c0, -c1, fw0, fw1, SLOT_Z[0], SLOT_Z[1]))
        out.append(_box("foot_n", look, fmat, -(t0 + TAB_THICK), -t0, fw0, fw1, SLOT_FLOOR - 1.0 * MM, SLOT_Z[1]))
        out.append(_box("foot_n", look, fmat, -t0, -c1, fw0, fw1, SLOT_FLOOR - 1.0 * MM, SLOT_FLOOR))
    return out


def rail_shapes() -> list[Shape]:
    L = RAIL_LENGTH / 2
    out = [_box("rail", "steel", "steel", -LIP_INNER, LIP_INNER, -L, L, 0.0, SHEET)]
    for s in (1, -1):
        out.append(_box("rail", "steel", "steel", *sorted((s * (LIP_INNER - SHEET), s * LIP_INNER)), -L, L,
                        SHEET, RAIL_HEIGHT))
        out.append(_box("rail", "steel", "steel", *sorted((s * LIP_INNER, s * LIP_EDGE)), -L, L,
                        RAIL_HEIGHT - LIP_THICK, RAIL_HEIGHT))
    top = BRACKET_HALF_X * 12 / 22
    for y0, y1 in (LEFT_BRACKET_Y, RIGHT_BRACKET_Y):
        # an end bracket: a tapered clamp standing on the lips, and a filler in the channel
        out.append(_prism("rail", "zinc", "steel", ((-BRACKET_HALF_X, RAIL_HEIGHT), (BRACKET_HALF_X, RAIL_HEIGHT),
                                                     (top, BRACKET_TOP), (-top, BRACKET_TOP)), y0, y1))
        out.append(_box("rail", "zinc", "steel", -(LIP_INNER - SHEET) + 0.5 * MM, LIP_INNER - SHEET - 0.5 * MM,
                        y0, y1, SHEET, RAIL_HEIGHT))
    if not RAIL_BEIGE:
        return out
    # the beige block, mounted against the left bracket: its housing collides, its feet are looks only
    for sh in sym_shapes("rail", "beige", (BEIGE_Y[0] + BEIGE_Y[1]) / 2, feet="visual",
                         ):
        out.append(_lift(sh, RAIL_HEIGHT + SEATED_HEIGHT))
    return out


def _lift(sh: Shape, dz: float) -> Shape:
    """A shape moved up by ``dz`` (block frame to rail frame)."""
    if sh.profile is not None:
        return Shape(sh.body, sh.look, sh.material, profile=tuple((x, z + dz) for x, z in sh.profile), span=sh.span,
                     visible=sh.visible)
    c = (sh.center[0], sh.center[1], sh.center[2] + dz)
    return Shape(sh.body, sh.look, sh.material, c, sh.size, None, sh.span, sh.radius, sh.length, sh.axis, sh.visible)


def block_shapes(look: str = "blue") -> list[Shape]:
    return (hook_shapes if BLOCK_MODEL == "HOOK" else sym_shapes)("body", look, 0.0)


# terminal blocks: the same jointed part in different colours (kind -> look)
BLOCK_KINDS = {"block": "blue", "block_grey": "grey", "block_beige": "beige"}
RAIL_BEIGE = True  # the rail carries a fixed beige block against its left bracket (False: a bare rail)


def tray_shapes() -> list[Shape]:
    ix, iy = TRAY_INNER[0] / 2, TRAY_INNER[1] / 2
    ox, oy = ix + TRAY_WALL, iy + TRAY_WALL
    top = TRAY_FLOOR + TRAY_WALL_HEIGHT
    return [
        _box("tray", "tray", "plastic", -ox, ox, -oy, oy, 0.0, TRAY_FLOOR),
        _box("tray", "tray", "plastic", -ox, -ix, -oy, oy, TRAY_FLOOR, top),
        _box("tray", "tray", "plastic", ix, ox, -oy, oy, TRAY_FLOOR, top),
        _box("tray", "tray", "plastic", -ix, ix, -oy, -iy, TRAY_FLOOR, top),
        _box("tray", "tray", "plastic", -ix, ix, iy, oy, TRAY_FLOOR, top),
    ]


# -- screwdriver and its holder -------------------------------------------------------------------------------------
# A flat-blade screwdriver (origin at the blade's tip, its axis along z): blade 3.5 x 0.7 mm, 10 mm long; shaft 5 mm
# square (a round look), 65 mm; handle 20 mm square (a round look), 100 mm.
BLADE = (0.7 * MM, 3.5 * MM, 10 * MM)
SHAFT = (5.0 * MM, 75 * MM)
HANDLE = (20.0 * MM, 100 * MM)
SCREWDRIVER_MASS = 0.06
HOLDER = (40 * MM, 9 * MM, 50 * MM, 5 * MM)  # outer side, hole side, height, floor thickness


def screwdriver_shapes() -> list[Shape]:
    bx, by = BLADE[0] / 2, BLADE[1] / 2
    h = SHAFT[0] / 2
    r = HANDLE[0] / 2
    return [
        _box("screwdriver", "steel", "steel", -bx, bx, -by, by, 0.0, BLADE[2]),
        _box("screwdriver", "steel", "steel", -h, h, -h, h, BLADE[2], SHAFT[1]),
        _box("screwdriver", "handle", "rubber", -r, r, -r, r, SHAFT[1], SHAFT[1] + HANDLE[1]),
    ]


# -- fuse -----------------------------------------------------------------------------------------------------------
# A 5 x 20 mm glass cartridge fuse (origin at its centre, its axis along x): a glass tube between two metal end caps.
# Collision: a 5 mm square bar (hidden): it does not roll, and the fingers grip flat faces (a cylinder collider
# slipped out of the fingertips).
FUSE = (5.0 * MM, 20.0 * MM)  # diameter, length
FUSE_CAP = 5.0 * MM  # each end cap's length
FUSE_MASS = 0.001


def fuse_shapes() -> list[Shape]:
    r, h = FUSE[0] / 2, FUSE[1] / 2
    bar = _box("fuse", "glass", "glass", -h, h, -r, r, -r, r)
    return [
        Shape(bar.body, bar.look, bar.material, bar.center, bar.size, visible=False),
        Shape("fuse", "glass", None, (0.0, 0.0, 0.0), radius=r * 0.94, length=FUSE[1] - 2 * FUSE_CAP, axis="X"),
        Shape("fuse", "screw", None, (-(h - FUSE_CAP / 2), 0.0, 0.0), radius=r, length=FUSE_CAP, axis="X"),
        Shape("fuse", "screw", None, (h - FUSE_CAP / 2, 0.0, 0.0), radius=r, length=FUSE_CAP, axis="X"),
    ]


def holder_shapes() -> list[Shape]:
    o, i, hgt, fl = HOLDER[0] / 2, HOLDER[1] / 2, HOLDER[2], HOLDER[3]
    return [
        _box("holder", "tray", "plastic", -o, o, -o, o, 0.0, fl),
        _box("holder", "tray", "plastic", -o, -i, -o, o, fl, hgt),
        _box("holder", "tray", "plastic", i, o, -o, o, fl, hgt),
        _box("holder", "tray", "plastic", -i, i, -o, -i, fl, hgt),
        _box("holder", "tray", "plastic", -i, i, i, o, fl, hgt),
    ]


FIXED_KINDS: set[str] = set()  # further fixed, single-body kinds a task adds (with their shapes in SHAPES)
FREE_KINDS: dict[str, float] = {}  # further free, single-body kinds a task adds: kind -> mass (kg)
# Functions run once per control step, before the scorer reads the states: f(base, states) (a task's own records)
STEP_HOOKS: list = []
_HOOK_STEP: dict = {"counter": None}
SHAPES = {"rail": rail_shapes, "tray": tray_shapes, "screwdriver": screwdriver_shapes, "holder": holder_shapes,
          "fuse": fuse_shapes,
          **{k: (lambda look=look: block_shapes(look)) for k, look in BLOCK_KINDS.items()}}


@lru_cache(maxsize=None)
def bounds(kind: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(low, high) corners of a part's geometry in its own frame."""
    lo, hi = [math.inf] * 3, [-math.inf] * 3
    for sh in SHAPES[kind]():
        for p in sh.points():
            for i in range(3):
                lo[i], hi[i] = min(lo[i], p[i]), max(hi[i], p[i])
    return tuple(lo), tuple(hi)


@lru_cache(maxsize=None)
def local_center(kind: str) -> tuple[float, float, float]:
    """The centre of a part's bounding box in its own frame: ``ObjectState.pos`` is this point."""
    lo, hi = bounds(kind)
    return tuple((a + b) / 2 for a, b in zip(lo, hi, strict=True))


def _triangulate(poly) -> list[tuple[int, int, int]]:
    """Ear-clipping triangulation of a simple counter-clockwise polygon [(x, z), ...]."""
    idx = list(range(len(poly)))
    tris = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def inside(p, a, b, c):
        return cross(a, b, p) >= 0 and cross(b, c, p) >= 0 and cross(c, a, p) >= 0

    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if cross(a, b, c) <= 1e-12:
                continue
            if any(inside(poly[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            raise ValueError("polygon is not simple or not counter-clockwise")
    tris.append(tuple(idx))
    return tris


# -- task-file objects --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Part(task_file.Box):
    """One of this module's parts as a task-file object: ``pos`` is the part's origin (see the module docstring),
    ``size`` its bounding box. Spawned by :func:`arena_object` once :func:`install` has run. A block with
    ``latched`` starts with its feet spread to their seated position (spawn it seated on a rail)."""

    kind: str = "block"
    latched: bool = False
    tilt: float = 0.0  # rad, about the part's own y axis (along the rail) after its yaw: > 0 lifts its -x end

    def _check(self) -> None:
        super()._check()
        if self.kind not in SHAPES:
            raise ValueError(f"{self.name}: unknown part kind {self.kind!r}")


def part(name: str, kind: str, pos, yaw: float = 0.0, fixed: bool = False, latched: bool = False,
         tilt: float = 0.0) -> Part:
    lo, hi = bounds(kind)
    if kind in BLOCK_KINDS:
        mass, color = (None if fixed else BLOCK_MASS + 2 * FOOT_MASS), LOOKS[BLOCK_KINDS[kind]][0]
    else:
        mass = None if fixed or kind in FIXED_KINDS else {"rail": RAIL_MASS, "tray": None, "holder": None, "screwdriver": SCREWDRIVER_MASS,
                                      "fuse": FUSE_MASS, **FREE_KINDS}[kind]
        color = {"rail": LOOKS["steel"][0], "tray": LOOKS["tray"][0], "holder": LOOKS["tray"][0],
                 "screwdriver": LOOKS["handle"][0], "fuse": LOOKS["glass"][0]}.get(kind, LOOKS["grey"][0])
    size = tuple(b - a for a, b in zip(lo, hi, strict=True))
    return Part(name=name, pos=tuple(pos), size=size, mass=mass, color=color, yaw=yaw, kind=kind, latched=latched,
                tilt=tilt)


@dataclass(frozen=True)
class BlockState(task_file.ObjectState):
    """A block's ``ObjectState`` plus, for its +x and -x side, in m: how far each foot is spread outward from rest
    (its tooth clears the lip edge at 2.0 mm), and how far each seat pad is squeezed in (latched: 0.9 mm)."""

    feet_spread: tuple[float, float] = (0.0, 0.0)
    pads_squeeze: tuple[float, float] = (0.0, 0.0)


# -- USD (after the simulator has started) ------------------------------------------------------------------

_USD_DIR: Path | None = None
_JOINTED: dict[str, str] = {}  # articulated parts spawned so far: name -> kind
# Parts defined outside this module: kind -> (write_usd(path), arena_object(spec), state(base, name)); a custom part
# may also define step(base, name, states), run once per control step before its state is read.
CUSTOM_KINDS: dict[str, tuple] = {}
_CUSTOM: dict[str, str] = {}  # custom parts spawned so far: name -> kind


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _apply(prim, schema: str, attrs: dict) -> None:
    """Apply a PhysX API schema by name and author its attributes (the PhysxSchema module isn't importable)."""
    from pxr import Sdf

    prim.AddAppliedSchema(schema)
    types = {bool: Sdf.ValueTypeNames.Bool, int: Sdf.ValueTypeNames.Int, float: Sdf.ValueTypeNames.Float,
             str: Sdf.ValueTypeNames.Token}
    for name, value in attrs.items():
        prim.CreateAttribute(name, types[type(value)]).Set(value)


def _prism_mesh(sh: Shape):
    """(points, face counts, face indices) of a prism: the profile at both ends of its span, caps and sides."""
    prof = list(sh.profile)
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(prof, prof[1:] + prof[:1]))
    if area < 0:
        prof = prof[::-1]
    n = len(prof)
    y0, y1 = sh.span
    pts = [(x, y0, z) for x, z in prof] + [(x, y1, z) for x, z in prof]
    if sh.axis == "X":  # a prism along x: the profile is in (y, z), the span along x
        pts = [(s, u, z) for u, s, z in pts]
    counts, indices = [], []
    # the profile is counter-clockwise in (x, z); seen from -y that is clockwise, so the y0 cap is reversed
    for tri in _triangulate(prof):
        counts.append(3)
        indices += [tri[0], tri[1], tri[2]]
        counts.append(3)
        indices += [n + tri[2], n + tri[1], n + tri[0]]
    for i in range(n):
        j = (i + 1) % n
        counts.append(4)
        indices += [i, n + i, n + j, j]
    return pts, counts, indices


def _new_stage(path: Path):
    """A new part file: (stage, its /Part root prim, looks by name, physics materials by name)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Part").GetPrim()
    stage.SetDefaultPrim(root)

    looks, mats = {}, {}
    for name, (rgb, rough, metal) in LOOKS.items():
        mat = UsdShade.Material.Define(stage, f"/Part/Looks/{name}")
        shader = UsdShade.Shader.Define(stage, f"/Part/Looks/{name}/shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(_srgb_to_linear, rgb)))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        looks[name] = mat
    for name, (static, dynamic, combine) in MATERIALS.items():
        mat = UsdShade.Material.Define(stage, f"/Part/PhysicsMaterials/{name}")
        api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr(static)
        api.CreateDynamicFrictionAttr(dynamic)
        api.CreateRestitutionAttr(0.0)
        _apply(mat.GetPrim(), "PhysxMaterialAPI", {"physxMaterial:frictionCombineMode": combine})
        mats[name] = mat
    return stage, root, looks, mats


def _xform(stage, path: str):
    from pxr import UsdGeom

    return UsdGeom.Xform.Define(stage, path).GetPrim()


def _write_shapes(stage, kind: str, body_paths: dict, looks: dict, mats: dict) -> None:
    """Every shape of a part, under its body's prim: looks, colliders and their physics materials."""
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, Vt

    for i, sh in enumerate(SHAPES[kind]()):
        parent = body_paths[sh.body]
        if sh.profile is not None:
            geom = UsdGeom.Mesh.Define(stage, f"{parent}/prism_{i}")
            pts, counts, indices = _prism_mesh(sh)
            geom.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
            geom.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
            geom.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
            geom.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            geom.CreateDoubleSidedAttr(True)
        elif sh.size is not None:
            geom = UsdGeom.Cube.Define(stage, f"{parent}/box_{i}")
            geom.CreateSizeAttr(1.0)
            geom.AddTranslateOp().Set(Gf.Vec3d(*sh.center))
            geom.AddScaleOp().Set(Gf.Vec3f(*sh.size))
        else:
            geom = UsdGeom.Cylinder.Define(stage, f"{parent}/cyl_{i}")
            geom.CreateRadiusAttr(sh.radius)
            geom.CreateHeightAttr(sh.length)
            geom.CreateAxisAttr(sh.axis)
            geom.AddTranslateOp().Set(Gf.Vec3d(*sh.center))
        prim = geom.GetPrim()
        if not sh.visible:
            UsdGeom.Imageable(prim).MakeInvisible()
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(looks[sh.look])
        if sh.material is None:
            continue
        UsdPhysics.CollisionAPI.Apply(prim)
        if sh.profile is not None:
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
        UsdShade.MaterialBindingAPI(prim).Bind(mats[sh.material], UsdShade.Tokens.strongerThanDescendants, "physics")
        _apply(prim, "PhysxCollisionAPI", {"physxCollision:contactOffset": CONTACT_OFFSET,
                                           "physxCollision:restOffset": 0.0})


def write_usd(kind: str, path: Path, fixed: bool = False) -> Path:
    """Write one part as a USD file: its bodies, colliders, looks, materials, and (for a block) the sprung feet."""
    from pxr import Gf, Sdf, UsdPhysics

    if kind in CUSTOM_KINDS:
        CUSTOM_KINDS[kind][0](path)
        return path
    stage, root, looks, mats = _new_stage(path)
    is_block = kind in BLOCK_KINDS
    bodies = ["body", "foot_p", "foot_n", "pad_p", "pad_n"] if is_block else {"rail": ["rail"], "tray": ["tray"], "holder": ["holder"], "screwdriver": ["screwdriver"], "fuse": ["fuse"]}.get(kind, [kind])
    masses = {**FREE_KINDS, "rail": RAIL_MASS, "screwdriver": SCREWDRIVER_MASS, "fuse": FUSE_MASS, "body": BLOCK_MASS, "foot_p": FOOT_MASS, "foot_n": FOOT_MASS, "pad_p": PAD_MASS,
              "pad_n": PAD_MASS}
    body_paths = {}
    for body in bodies:
        # the rail and the tray are the root prim itself; a block's links are its children
        path_b = "/Part" if not is_block else f"/Part/{body}"
        prim = root if not is_block else _xform(stage, path_b)
        body_paths[body] = path_b
        if kind in ("tray", "holder") or kind in FIXED_KINDS or fixed:
            continue  # a fixed collider: no rigid body
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(masses[body])
        _apply(prim, "PhysxRigidBodyAPI", {
            "physxRigidBody:solverPositionIterationCount": 32,
            "physxRigidBody:solverVelocityIterationCount": 4,
            "physxRigidBody:maxDepenetrationVelocity": 0.3,
            "physxRigidBody:enableCCD": False,
        })
        if body == "body":  # contact sensing: the robot's fingertips gripping the block
            _apply(prim, "PhysxContactReportAPI", {"physxContactReport:threshold": 0.0})
    _write_shapes(stage, kind, body_paths, looks, mats)

    if is_block:
        UsdPhysics.ArticulationRootAPI.Apply(root)
        _apply(root, "PhysxArticulationAPI", {
            "physxArticulation:enabledSelfCollisions": False,
            "physxArticulation:solverPositionIterationCount": 32,
            "physxArticulation:solverVelocityIterationCount": 4,
        })
        snap = SNAP_OFF_DEG is not None
        hooked = snap or FIXED_HOOK
        joints = (  # name, parent, child, axis, limits, spring, damping
            # snap-off variant: the +x foot is a fixed hook (no travel); the -x foot slides on an inclined slide
            # the fixed hook: locked, or (HOOK_FLEX) a stiff spring standing in for the plastic hook's flex
            ("foot_p", "body", "foot_p", "X", ((0.0, HOOK_FLEX) if HOOK_FLEX else (0.0, 0.0)) if hooked
             else (0.0, FOOT_TRAVEL), HOOK_SPRING if hooked else FOOT_SPRING, FOOT_DAMPING),
            ("foot_n", "body", "foot_n", "X",
             (-FOOT_TRAVEL / math.cos(math.radians(SNAP_OFF_DEG)) if snap else -FOOT_TRAVEL, 0.0),
             SNAP_SPRING if snap else FOOT_SPRING, FOOT_DAMPING),
            ("pad_p", "body", "pad_p", "Z", (0.0, PAD_PROUD), PAD_SPRING, PAD_DAMPING),
            ("pad_n", "body", "pad_n", "Z", (0.0, PAD_PROUD), PAD_SPRING, PAD_DAMPING),
        )
        for name, parent, child, axis, (lo, hi), k, c in joints:
            joint = UsdPhysics.PrismaticJoint.Define(stage, f"/Part/joints/{name}")
            joint.CreateBody0Rel().SetTargets([Sdf.Path(body_paths[parent])])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(body_paths[child])])
            joint.CreateAxisAttr(axis)
            joint.CreateLowerLimitAttr(lo)
            joint.CreateUpperLimitAttr(hi)
            joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            rot = Gf.Quatf(1, 0, 0, 0)
            if snap and name == "foot_n":
                # the slide's axis tilted SNAP_OFF_DEG up toward the rail's centre: the foot moves out and down
                # (negative travel) when the lip pushes its tooth down, i.e. when the block is pulled up
                half = -math.radians(SNAP_OFF_DEG) / 2
                rot = Gf.Quatf(math.cos(half), 0.0, math.sin(half), 0.0)
            joint.CreateLocalRot0Attr(rot)
            joint.CreateLocalRot1Attr(rot)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(k)
            drive.CreateDampingAttr(c)
            drive.CreateTargetPositionAttr(0.0)
            _apply(joint.GetPrim(), "PhysxJointAPI",
                   {"physxJoint:armature": FOOT_ARMATURE if name.startswith("foot") else PAD_ARMATURE})
    stage.GetRootLayer().Save()
    return path


def usd_file(kind: str, fixed: bool = False) -> str:
    global _USD_DIR
    if _USD_DIR is None:
        _USD_DIR = Path(tempfile.mkdtemp(prefix="terminal_block_parts_"))
    path = _USD_DIR / f"{kind}{'_fixed' if fixed else ''}.usda"
    if not path.exists():
        write_usd(kind, path, fixed)
    return str(path)


def arena_object(spec: Part):
    """The Arena object for a part: a block an articulation, a rail with a mass a free rigid body, the rest fixed."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    squeeze = PAD_PROUD - SEATED_HEIGHT if spec.latched else 0.0

    class _Block(Object):
        """A block: its feet's joint drives are the springs, written to PhysX by implicit actuators."""

        def _generate_articulation_cfg(self):
            cfg = super()._generate_articulation_cfg()
            cfg.actuators = {
                "feet": ImplicitActuatorCfg(joint_names_expr=["foot_n"] if FIXED_HOOK else ["foot_.*"],
                                            stiffness=FOOT_SPRING, damping=FOOT_DAMPING, armature=FOOT_ARMATURE),
                **({"hook": ImplicitActuatorCfg(joint_names_expr=["foot_p"], stiffness=HOOK_SPRING, damping=FOOT_DAMPING,
                                                armature=FOOT_ARMATURE)} if FIXED_HOOK else {}),
                "pads": ImplicitActuatorCfg(joint_names_expr=["pad_.*"], stiffness=PAD_SPRING, damping=PAD_DAMPING,
                                            armature=PAD_ARMATURE),
            }
            cfg.init_state.joint_pos = {"foot_.*": 0.0, "pad_.*": squeeze}
            return cfg

    if spec.kind in CUSTOM_KINDS:
        _CUSTOM[spec.name] = spec.kind
        return CUSTOM_KINDS[spec.kind][1](spec)
    cz, sz, cy, sy = math.cos(spec.yaw / 2), math.sin(spec.yaw / 2), math.cos(spec.tilt / 2), math.sin(spec.tilt / 2)
    # yaw about z, then tilt about the part's own y: q = qz(yaw) * qy(tilt)
    pose = Pose(position_xyz=tuple(spec.pos), rotation_xyzw=(-sz * sy, cz * sy, sz * cy, cz * cy))
    fixed = spec.mass is None
    cfg = sim_utils.UsdFileCfg(usd_path=usd_file(spec.kind, fixed and spec.kind == "rail"))
    if spec.kind in BLOCK_KINDS:
        _JOINTED[spec.name] = spec.kind
        return _Block(name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=cfg, initial_pose=pose)
    kind = ObjectType.RIGID if (spec.kind in ("rail", "screwdriver", "fuse") or spec.kind in FREE_KINDS) and not fixed \
        else ObjectType.BASE
    return Object(name=spec.name, object_type=kind, spawner_cfg=cfg, initial_pose=pose)


# -- the hook into yam_arena.scene --------------------------------------------------------------------------


_MOTION: dict = {}  # name -> (control step, root position, rotation, velocities over the step that ended there)


def _motion(base, name: str, pos, rot):
    """A jointed part's velocities (m/s, rad/s) from its root's pose change over the last control step, or
    (None, None) on an episode's first step. Clamped by its seat pads, a block gets a steady velocity from PhysX
    that it does not have (see PAD_ARMATURE); its real motion is what the scorer's "still" check is about."""
    import numpy as np

    counter = getattr(base, "common_step_counter", None)
    episode = getattr(base, "episode_length_buf", None)
    if counter is None or episode is None or int(episode[0]) == 0:
        _MOTION.pop(name, None)
        return None, None
    last = _MOTION.get(name)
    if last is not None and last[0] == counter:  # asked again within the same step
        return last[3], last[4]
    if last is None or last[0] != counter - 1:
        _MOTION[name] = (counter, pos, rot, None, None)
        return None, None
    dt = float(base.step_dt)
    lin = (pos - last[1]) / dt
    d = rot @ last[2].T  # rotation over the step
    angle = math.acos(min(1.0, max(-1.0, (np.trace(d) - 1.0) / 2.0)))
    axis = np.array([d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]])
    n = np.linalg.norm(axis)
    ang = axis / n * angle / dt if n > 1e-12 else np.zeros(3)
    _MOTION[name] = (counter, pos, rot, lin, ang)
    return lin, ang


def _articulation_state(base, name: str, kind: str) -> BlockState:
    """A jointed part's state, as ``yam_arena.scene.object_states`` reports rigid objects (bounding-box centre from
    this module's geometry, carried by the root link's pose; orientation; velocities over the last control step; bounds), plus its
    feet and seat pads."""
    import numpy as np

    art = base.scene.articulations[name]
    data = art.data
    pose = data.root_pose_w.torch[0].detach().cpu().numpy().astype(np.float64)
    origin = base.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
    pos, quat = pose[:3] - origin, pose[3:]
    x, y, z, w = quat
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    lo, hi = bounds(kind)
    corners = np.array([[(lo, hi)[(i >> k) & 1][k] for k in range(3)] for i in range(8)]) @ rot.T + pos
    q = data.joint_pos.torch[0].detach().cpu().numpy().astype(np.float64)
    names = list(art.joint_names)
    lin_vel, ang_vel = _motion(base, name, pos, rot)
    if lin_vel is None:  # first step of an episode: PhysX's figures (see PAD_ARMATURE: they may read high)
        lin_vel = data.root_lin_vel_w.torch[0].detach().cpu().numpy().astype(np.float64)
        ang_vel = data.root_ang_vel_w.torch[0].detach().cpu().numpy().astype(np.float64)
    return BlockState(
        pos=rot @ np.array(local_center(kind)) + pos,
        quat_xyzw=quat,
        lin_vel=lin_vel,
        ang_vel=ang_vel,
        low=corners.min(axis=0),
        high=corners.max(axis=0),
        feet_spread=(float(q[names.index("foot_p")]), float(-q[names.index("foot_n")])),
        pads_squeeze=(float(q[names.index("pad_p")]), float(q[names.index("pad_n")])),
    )


def _patch(scene_module) -> None:
    original = getattr(scene_module, "task_object", None)
    original_states = getattr(scene_module, "object_states", None)
    # fail loudly if yam_arena changes the two functions this hook wraps
    for fn, params in ((original, ["spec"]), (original_states, ["env", "names"])):
        if not callable(fn) or list(inspect.signature(fn).parameters)[: len(params)] != params:
            raise RuntimeError(
                "tb_parts: yam_arena.scene no longer has task_object(spec) and object_states(env, names); "
                "this task's parts hook needs updating for this yam_arena version"
            )
    if getattr(original, "_spawns_parts", False):
        return

    def task_object(spec):
        return arena_object(spec) if isinstance(spec, Part) else original(spec)

    def object_states(env, names):
        # Arena's geometry queries don't cover articulations, so a block's state is built here.
        base = env.unwrapped
        jointed = {n for n in names if n in _JOINTED and n in base.scene.articulations}
        custom = {n for n in names if n in _CUSTOM}
        out = original_states(env, [n for n in names if n not in jointed | custom])
        out.update({n: _articulation_state(base, n, _JOINTED[n]) for n in jointed})
        for n in sorted(custom):  # a custom part's step (a scripted hand) runs once per control step
            entry = CUSTOM_KINDS[_CUSTOM[n]]
            if len(entry) > 3:
                entry[3](base, n, out)
            out[n] = entry[2](base, n)
        if STEP_HOOKS:  # a task's own per-step records, once per control step (object_states may be asked again)
            key = (int(base.episode_length_buf[0]), int(getattr(base, "common_step_counter", 0)))
            if key != _HOOK_STEP["counter"]:
                _HOOK_STEP["counter"] = key
                for hook in STEP_HOOKS:
                    hook(base, out)
        return {n: out[n] for n in names}

    task_object._spawns_parts = True
    task_object.__doc__ = original.__doc__
    object_states.__doc__ = original_states.__doc__
    scene_module.task_object = task_object
    scene_module.object_states = object_states


class _SceneHook(importlib.abc.MetaPathFinder):
    """Patches ``yam_arena.scene`` right after it is first imported (after the simulator has started)."""

    def find_spec(self, fullname, path, target=None):
        if fullname != "yam_arena.scene":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


def install() -> None:
    """Make yam_arena spawn :class:`Part` objects. Safe to call more than once."""
    if "yam_arena.scene" in sys.modules:
        _patch(sys.modules["yam_arena.scene"])
    elif not any(isinstance(f, _SceneHook) for f in sys.meta_path):
        sys.meta_path.insert(0, _SceneHook())
