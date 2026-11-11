"""A mixed terminal strip with three fuse terminal blocks, each with a spare-fuse pocket, and a rack that presents
one spare fuse upright.

The strip is one fixed part (its blocks are latched on the rail: they do not move). From the left: two blue and one
grey plain feed-through block, three grey fuse blocks, two beige and one grey plain block. A fuse block is longer and
taller than the plain ones, carries an upright fuse-holder plate on the far half of its top and, in the near half of
its top, a square spare-fuse pocket: a 5 x 20 mm cartridge fuse stands in it, 6 mm of it out of the top.

Frames: the strip's origin is on the rail's lip-top plane, on the rail's centreline, at the middle of the middle
fuse block; x across the rail (away from the arms), y along it (the robot's left), z up. The rack's origin is the
centre of its underside. The fuse is a separate, free part (``tb_parts`` kind "fuse").
"""

from __future__ import annotations

import tb_parts as tp

MM = 1e-3
STRIP, RACK = "fuse_strip", "fuse_rack"

# -- the strip (strip frame, metres) ----------------------------------------------------------------------------
SEAM = 0.15 * MM  # each plain housing drawn this much short of its faces, so the blocks read as separate
FUSE_WIDTH = 8.2 * MM  # a fuse block's thickness along the rail
TOP = 34 * MM  # a fuse block's top above the lip tops
BLOCKS = (  # from the left: (kind, look, thickness along the rail)
    ("plain", "blue", 5.1 * MM), ("plain", "blue", 5.1 * MM), ("plain", "grey", 6.2 * MM),
    ("fuse", "grey", FUSE_WIDTH), ("fuse", "grey", FUSE_WIDTH), ("fuse", "grey", FUSE_WIDTH),
    ("plain", "beige", 5.1 * MM), ("plain", "beige", 5.1 * MM), ("plain", "grey", 6.2 * MM),
)
FUSE_BLOCKS = tuple(i for i, b in enumerate(BLOCKS) if b[0] == "fuse")  # indices into BLOCKS, from the left
PLAIN_PROFILE = [(-26, -4.0), (-23, -6.0), (-19, -6.0), (-18.2, 0.3), (18.2, 0.3), (19, -6.0), (23, -6.0), (26, -4.0),
                 (26, 17), (22, 17), (22, 22), (15, 22), (15, 28), (-15, 28), (-15, 22), (-22, 22), (-22, 17), (-26, 17)]
PLAIN_BOXES = [(-26, 26, 0.3, 17), (-22, 22, 17, 22), (-15, 15, 22, 28)]  # (x0, x1, z0, z1), mm
POCKET_X = -14.5 * MM  # the pocket's axis, across the rail
BORE = 6.0 * MM  # the pocket's square bore (the fuse is 5 mm): a loose fit, as a storage pocket's
DEPTH = 14 * MM  # below the block's top: a stored fuse stands 6 mm out
LEAD_IN = 2.0 * MM  # 45 degree lead-in at the pocket's mouth, on its near and far edges ...
LEAD_IN_SIDE = 0.9 * MM  # ... and on its two side edges (the side walls are 0.95 mm thick)
HOLDER = (-5 * MM, 20 * MM, 24 * MM, 6.0 * MM)  # the fuse-holder plate: x0, x1, height above the top, thickness


def block_y(i: int) -> float:
    """Centre of block i (0 = leftmost) along the rail; the middle fuse block is at 0."""
    widths = [b[2] for b in BLOCKS]
    mid = FUSE_BLOCKS[len(FUSE_BLOCKS) // 2]
    left_edge = sum(widths[:mid]) + widths[mid] / 2  # the strip's left end, from the middle fuse block's centre
    return left_edge - sum(widths[:i]) - widths[i] / 2


def pocket(k: int) -> tuple[float, float, float, float]:
    """The pocket of fuse block k (0 = the leftmost fuse block): its axis (x, y), its mouth's and floor's z."""
    return POCKET_X, block_y(FUSE_BLOCKS[k]), TOP, TOP - DEPTH


def strip_shapes() -> list[tp.Shape]:
    out = []

    def hidden(b):
        return tp.Shape(b.body, b.look, b.material, b.center, b.size, visible=False)

    for i, (kind, look, w) in enumerate(BLOCKS):
        y = block_y(i)
        y0, y1 = y - w / 2, y + w / 2
        if kind == "plain":
            out.append(tp._prism(STRIP, look, None, [(x * MM, z * MM) for x, z in PLAIN_PROFILE], y0 + SEAM, y1 - SEAM))
            out += [hidden(tp._box(STRIP, look, "polyamide", x0 * MM, x1 * MM, y0, y1, z0 * MM, z1 * MM))
                    for x0, x1, z0, z1 in PLAIN_BOXES]
            continue
        box = lambda *v: tp._box(STRIP, look, "polyamide", *v)  # noqa: E731
        s0, s1 = y0 + SEAM, y1 - SEAM  # drawn (and collided) just short of its faces

        def side(*profile, look=look):  # a side lead-in's slope: a prism along x, its profile in (y, z)
            return tp.Shape(STRIP, look, "polyamide", profile=tuple(profile), span=(POCKET_X - BORE / 2 - LEAD_IN,
                                                                                    POCKET_X + BORE / 2 + LEAD_IN), axis="X")
        h0, h1 = POCKET_X - BORE / 2, POCKET_X + BORE / 2
        b0, b1 = y - BORE / 2, y + BORE / 2
        floor, lead = TOP - DEPTH, TOP - LEAD_IN
        out += [
            box(-31 * MM, 31 * MM, s0, s1, 0.3 * MM, floor),  # below the pocket (its top is the pocket's floor)
            box(-31 * MM, -27 * MM, s0, s1, -4 * MM, 0.3 * MM), box(27 * MM, 31 * MM, s0, s1, -4 * MM, 0.3 * MM),  # feet
            box(-27 * MM, h0, s0, s1, floor, 26 * MM), box(h1, 27 * MM, s0, s1, floor, 26 * MM),  # the step, either side
            box(-20 * MM, h0, s0, s1, 26 * MM, lead), box(h1, 20 * MM, s0, s1, 26 * MM, lead),  # the top, either side
            box(-20 * MM, h0 - LEAD_IN, s0, s1, lead, TOP), box(h1 + LEAD_IN, 20 * MM, s0, s1, lead, TOP),
            box(h0 - LEAD_IN, h1 + LEAD_IN, s0, b0, floor, TOP - LEAD_IN_SIDE),  # side walls, below their lead-ins
            box(h0 - LEAD_IN, h1 + LEAD_IN, b1, s1, floor, TOP - LEAD_IN_SIDE),
            box(h0 - LEAD_IN, h1 + LEAD_IN, s0, b0 - LEAD_IN_SIDE, TOP - LEAD_IN_SIDE, TOP),
            box(h0 - LEAD_IN, h1 + LEAD_IN, b1 + LEAD_IN_SIDE, s1, TOP - LEAD_IN_SIDE, TOP),
            side((b0 - LEAD_IN_SIDE, TOP), (b0 - LEAD_IN_SIDE, TOP - LEAD_IN_SIDE), (b0, TOP - LEAD_IN_SIDE)),
            side((b1, TOP - LEAD_IN_SIDE), (b1 + LEAD_IN_SIDE, TOP - LEAD_IN_SIDE), (b1 + LEAD_IN_SIDE, TOP)),
            # the lead-in's two slopes
            tp._prism(STRIP, look, "polyamide", [(h0 - LEAD_IN, TOP), (h0 - LEAD_IN, lead), (h0, lead)], b0, b1),
            tp._prism(STRIP, look, "polyamide", [(h1, lead), (h1 + LEAD_IN, lead), (h1 + LEAD_IN, TOP)], b0, b1),
            box(HOLDER[0], HOLDER[1], y - HOLDER[3] / 2, y + HOLDER[3] / 2, TOP, TOP + HOLDER[2]),  # the fuse holder
        ]
    return out


# -- the rack (rack frame) ---------------------------------------------------------------------------------------
RACK_SIZE = (30 * MM, 30 * MM, 12 * MM)
RACK_BORE = 6.6 * MM
RACK_DEPTH = 8 * MM  # the fuse stands 12 mm out of the rack


def rack_shapes() -> list[tp.Shape]:
    hx, hy, h = RACK_SIZE[0] / 2, RACK_SIZE[1] / 2, RACK_SIZE[2]
    b, f = RACK_BORE / 2, RACK_SIZE[2] - RACK_DEPTH
    box = lambda *v: tp._box(RACK, "tray", "plastic", *v)  # noqa: E731
    return [box(-hx, hx, -hy, hy, 0.0, f), box(-hx, -b, -hy, hy, f, h), box(b, hx, -hy, hy, f, h),
            box(-b, b, -hy, -b, f, h), box(-b, b, b, hy, f, h)]


FUSE_IN_RACK = (0.0, 0.0, RACK_SIZE[2] - RACK_DEPTH + tp.FUSE[1] / 2)  # the fuse's centre, standing in the rack


def install() -> None:
    tp.SHAPES[STRIP] = strip_shapes
    tp.SHAPES[RACK] = rack_shapes
    tp.FIXED_KINDS.update((STRIP, RACK))
