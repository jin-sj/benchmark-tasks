"""Printable parts of the task's terminal block, generated from the same constants as the simulation.

    python build/make_latch.py          # writes housing.stl, foot.stl and latch_drawing.png into build/

Every latch dimension comes from ``tb_parts.py`` as configured by ``task.py`` (the simulated block). The one
difference is the rail: the simulation's lips are 2 mm thick (a 1 mm lip lets parts tunnel through at 240 Hz), a
real TS35 x 7.5 rail's 1 mm, so the printed feet's teeth sit 1 mm higher, keeping the same 0.3 mm under the lip.

Both parts are 2.5D: outlines in the block's section (x across the rail, z up) extruded along the rail (y) in a few
layers. They print lying on a broad face, the layers being the printer's layers (the STL's Z is the block's y).
Units: mm. The STLs hold one closed shell per layer; the layers touch face to face, which slicers merge.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import task  # noqa: E402,F401  (configures tb_parts as the task uses it)
import tb_parts as tp  # noqa: E402

REAL_LIP = 1.0  # mm: a real TS35 rail's lip thickness
MM = 1e3  # m -> mm


def mm(v: float) -> float:
    return v * MM


def latch_profile(reach: float | None) -> dict:
    """The foot/hook collision profile pieces (u, z mm) for a real 1 mm lip."""
    saved = tp._LIP_Z
    tp._LIP_Z = -REAL_LIP
    try:
        return tp._foot_profile(False, reach)
    finally:
        tp._LIP_Z = saved


def design(points) -> Polygon:
    """A HOOK design-frame outline (across, up) as a block-frame polygon in mm."""
    return Polygon([(x * MM, z * MM) for x, z in tp._hook_frame(points)])


def rect(x0, x1, z0, z1) -> Polygon:
    return box(min(x0, x1), min(z0, z1), max(x0, x1), max(z0, z1))


W = mm(tp.BLOCK_WIDTH) / 2  # housing half-width
F = mm(tp.FOOT_HALF_WIDTH)  # latch parts' half-width


def hw(v):
    return F if v is None else v


def housing_features() -> list[tuple[str, Polygon, float, float]]:
    """(op, section polygon, y0, y1): '+' adds material over y0..y1, '-' removes it."""
    out = []
    seat = mm(tp.SEATED_HEIGHT)
    body = design(tp._HOOK_BODY_LOOK[0]).intersection(rect(-100, 100, seat, 100))
    out.append(("+", body, -W, W))
    c0, c1 = mm(tp.HOOK_SLOT_CHEEK[0]), mm(tp.HOOK_SLOT_CHEEK[1])
    t0 = c0 + mm(tp.SLOT_WIDTH)
    z0, z1 = mm(tp.SLOT_Z[0]), mm(tp.SLOT_Z[1])
    out.append(("+", rect(-c0, -c1, z0, z1), -F, F))  # the cheek
    for s in (1, -1):  # the release slot's side walls
        y0, y1 = sorted((s * mm(tp.SLOT_LENGTH) / 2, s * W))
        out.append(("+", rect(-t0, -c0, z0, z1), y0, y1))
    hook = latch_profile(tp.HOOK_REACH)  # the fixed hook (+x)
    for key in ("hook", "arm"):
        out.append(("+", Polygon(hook[key]), -F, F))
    x0, x1, z0, z1, h = tp.BUILD_HOOK_WEB
    out.append(("+", rect(x0, x1, z0, z1), -hw(h), hw(h)))
    x0, x1, z0, z1, inner = tp.BUILD_GUIDE_WALLS
    px, pz, pd, _ = tp.BUILD_GUIDE_PIN
    for s in (1, -1):
        y0, y1 = sorted((s * inner, s * W))
        out.append(("+", rect(x0, x1, z0, z1), y0, y1))
        out.append(("-", Point(px, pz).buffer(pd / 2 - 0.03, 24), y0, y1))  # the pin's press fit
    root, tip, t, slit = tp.BUILD_PAD_LEAF
    px0, px1 = mm(tp.PAD_X[0]), mm(tp.PAD_X[1])
    lower = design(tp._HOOK_HULLS[1]).bounds  # the housing's lower middle part, over the rail
    for s, edge in ((1, lower[2]), (-1, lower[0])):  # the seat pads: a bump under a flexure leaf cut free of it
        out.append(("-", rect(s * root, edge + s * 0.01, seat + t, seat + t + slit), -W - 1, W + 1))
        out.append(("+", rect(s * root, s * tip, seat, seat + t), -W, W))
        out.append(("+", rect(s * px0, s * px1, -mm(tp.PAD_PROUD), seat), -F, F))
    x0, x1, z0, z1, h = tp.BUILD_SPRING_SEAT
    out.append(("-", rect(x0, x1, z0, z1), -h, h))
    depth, side = tp.BUILD_WIRE_ENTRY
    wz = mm(tp.WIRE_Z)
    for xf, s in ((mm(tp.BLOCK_NEAR_FACE), 1), (mm(tp.BLOCK_FAR_X), -1)):
        out.append(("-", rect(xf - s * 1.0, xf + s * depth, wz - side / 2, wz + side / 2), -side / 2, side / 2))
    return out


def foot_features() -> list[tuple[str, Polygon, float, float]]:
    out = []
    for pts in tp._HOOK_FOOT_LOOK:
        out.append(("+", design(pts), -F, F))
    prof = latch_profile(None)
    for key in ("hook", "arm"):
        out.append(("+", Polygon([(-u, z) for u, z in prof[key]]), -F, F))
    for u0, u1, z0, z1 in tp.FOOT_SOLE:
        out.append(("+", rect(-u0, -u1, z0, z1), -F, F))
    c0, c1 = mm(tp.HOOK_SLOT_CHEEK[0]), mm(tp.HOOK_SLOT_CHEEK[1])
    t0 = c0 + mm(tp.SLOT_WIDTH)
    floor = mm(tp.SLOT_FLOOR)
    out.append(("+", rect(-(t0 + mm(tp.TAB_THICK)), -t0, floor - 1.0, mm(tp.SLOT_Z[1])), -F, F))  # the tab
    out.append(("+", rect(-t0, -c1, floor - 1.0, floor), -F, F))  # the slot's floor
    for feat in (tp.BUILD_FOOT_WEB, tp.BUILD_FOOT_HEAD, tp.BUILD_FOOT_TONGUE, tp.BUILD_FOOT_BOSS):
        x0, x1, z0, z1, h = feat
        out.append(("+", rect(x0, x1, z0, z1), -hw(h), hw(h)))
    x0, x1, z0, z1 = tp.BUILD_TONGUE_SLOT
    r = (z1 - z0) / 2
    h = tp.BUILD_FOOT_TONGUE[4]
    out.append(("-", LineString([(x0 + r, z0 + r), (x1 - r, z0 + r)]).buffer(r, 24), -h - 1, h + 1))
    return out


def layers(features) -> list[tuple[float, float, Polygon]]:
    ys = sorted({round(v, 4) for _, _, y0, y1 in features for v in (y0, y1)})
    out = []
    for a, b in zip(ys[:-1], ys[1:]):
        mid = (a + b) / 2
        add = [p for op, p, y0, y1 in features if op == "+" and y0 <= mid <= y1]
        sub = [p for op, p, y0, y1 in features if op == "-" and y0 <= mid <= y1]
        if not add:
            continue
        poly = unary_union(add)
        if sub:
            poly = poly.difference(unary_union(sub))
        poly = shapely.set_precision(poly, 1e-4)
        if not poly.is_empty:
            out.append((a, b, poly))
    return out


def extrude(poly, y0: float, y1: float):
    """Closed triangle shells (vertices, faces) of a section polygon extruded over y0..y1, in print axes (x, z, y)."""
    import trimesh

    shells = []
    for part in getattr(poly, "geoms", [poly]):
        if part.is_empty or part.area < 1e-6:
            continue
        part = shapely.geometry.polygon.orient(part, 1.0)
        tris = shapely.constrained_delaunay_triangles(part)
        verts, faces = [], []

        def vid(x, z, y):
            verts.append((x, z, y))
            return len(verts) - 1

        for t in tris.geoms:
            (ax, az), (bx, bz), (cx, cz) = list(t.exterior.coords)[:3]
            faces.append((vid(ax, az, y0), vid(cx, cz, y0), vid(bx, bz, y0)))
            faces.append((vid(ax, az, y1), vid(bx, bz, y1), vid(cx, cz, y1)))
        for ring in [part.exterior, *part.interiors]:
            c = list(ring.coords)
            for (ax, az), (bx, bz) in zip(c[:-1], c[1:]):
                i = [vid(ax, az, y0), vid(bx, bz, y0), vid(bx, bz, y1), vid(ax, az, y1)]
                faces += [(i[0], i[1], i[2]), (i[0], i[2], i[3])]
        m = trimesh.Trimesh(np.array(verts), np.array(faces), process=True)
        trimesh.repair.fix_normals(m)
        shells.append(m)
    return shells


def write_stl(features, path: Path) -> dict:
    import trimesh

    shells = []
    for a, b, poly in layers(features):
        shells += extrude(poly, a + W, b + W)  # print axes: Z = y + W, on the bed at Z = 0
    mesh = trimesh.util.concatenate(shells)
    mesh.export(path)
    return {"shells": len(shells), "watertight_shells": sum(s.is_watertight for s in shells),
            "volume_mm3": round(sum(s.volume for s in shells), 1), "extent_mm": [round(v, 2) for v in mesh.extents]}


def drawing(path: Path) -> None:
    """Sections of the assembled latch at the block's middle (y 0) and through a guide wall, with the real rail."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def section(features, y):
        return layers([(op, p, y0, y1) for op, p, y0, y1 in features if y0 <= y <= y1] or [])

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    rail = [rect(-17.5, -13.5, -mm(tp.SEATED_HEIGHT) - REAL_LIP, -mm(tp.SEATED_HEIGHT)),
            rect(13.5, 17.5, -mm(tp.SEATED_HEIGHT) - REAL_LIP, -mm(tp.SEATED_HEIGHT))]
    for ax, y, title in ((axes[0], 0.0, "section at the block's middle (y 0)"),
                         (axes[1], (tp.BUILD_GUIDE_WALLS[4] + W) / 2, "section through a guide wall")):
        for feats, color in ((housing_features(), "#c9c2a6"), (foot_features(), "#5b8bd6")):
            for a, b, poly in layers(feats):
                if a <= y <= b:
                    for g in getattr(poly, "geoms", [poly]):
                        xs, zs = g.exterior.xy
                        ax.fill(xs, zs, color=color, ec="k", lw=0.5)
                        for hole in g.interiors:
                            hx, hz = hole.xy
                            ax.fill(hx, hz, color="white", ec="k", lw=0.5)
        for r in rail:
            xs, zs = r.exterior.xy
            ax.fill(xs, zs, color="#888", ec="k", lw=0.5)
        wire, od, free, axis_z = tp.BUILD_SPRING
        x0 = tp.BUILD_SPRING_SEAT[0]
        x1 = tp.BUILD_FOOT_HEAD[0]
        if y == 0.0:
            xs = np.linspace(x0, x1, 41)
            ax.plot(xs, axis_z + (od / 2 - wire / 2) * np.sign(np.sin(np.linspace(0, 8 * np.pi, 41))), color="#333", lw=1)
        px, pz, pd, _ = tp.BUILD_GUIDE_PIN
        ax.add_patch(plt.Circle((px, pz), pd / 2, color="#555"))
        ax.set_aspect("equal")
        ax.set_xlim(-40, 40)
        ax.set_ylim(-9, 31)
        ax.grid(True, lw=0.3)
        ax.set_title(title + "; mm, x across the rail (toward the arms negative), z up from the lip tops")
    fig.text(0.01, 0.01, "housing: beige; sprung foot: blue; rail lips (1 mm, real): grey; steel pin 1.5 x 6 mm; "
             "compression spring 0.3 x 3.0 x 5.6 mm, 1.0 N/mm", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> int:
    for name, feats in (("housing", housing_features()), ("foot", foot_features())):
        info = write_stl(feats, HERE / f"{name}.stl")
        print(name, info)
    drawing(HERE / "latch_drawing.png")
    print("drawing", HERE / "latch_drawing.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
