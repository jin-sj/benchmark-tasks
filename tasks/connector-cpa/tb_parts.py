"""Shared part helpers: shapes, USD writing, and the hook that lets yam_arena spawn a task's own parts.

yam_arena task files support single primitives and library assets only, and yam_arena has no extension point
for multi-shape or jointed parts. This module is the task's own minimal mechanism, kept in the task folder:

- :class:`Part` is a :class:`yam_arena.task_file.Box` (so the stock task-file loader accepts it and the
  scorer gets an ``ObjectState`` for it) that also names which part kind it is.
- :func:`install` wraps ``yam_arena.scene.task_object`` and ``object_states`` the moment that module is imported
  (it imports Isaac Sim, so it is never imported here before the simulator starts). Every :class:`Part` is then
  spawned from a USD file its kind writes; every other object goes through yam_arena unchanged.
- A task module registers its part kinds in ``CUSTOM_KINDS`` (write_usd, arena_object, state), their shapes in
  ``SHAPES``, and its looks and physics materials; ``STEP_HOOKS`` run once per control step.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import inspect
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from yam_arena import task_file

MM = 1e-3

# -- materials and looks ------------------------------------------------------------------------------------
# (static friction, dynamic friction, combine mode). PhysX uses the combine mode of higher priority
# (average < min < multiply < max).
MATERIALS = {
    "steel": (0.30, 0.30, "average"),
    "latch": (0.25, 0.25, "min"),  # plastic latch faces on plastic
    "plastic": (0.40, 0.40, "average"),
    "rubber": (0.90, 0.90, "average"),
}
LOOKS: dict = {}  # name -> (sRGB, roughness, metallic); a task adds its own
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


STEP_HOOKS: list = []
_HOOK_STEP: dict = {"counter": None}
SHAPES: dict = {}  # kind -> function returning its shapes; a task adds its own


@lru_cache(maxsize=None)
def bounds(kind: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(low, high) corners of a part's geometry in its own frame."""
    lo, hi = [float("inf")] * 3, [float("-inf")] * 3
    for sh in SHAPES[kind]():
        for p in sh.points():
            for i in range(3):
                lo[i], hi[i] = min(lo[i], p[i]), max(hi[i], p[i])
    return tuple(lo), tuple(hi)


@lru_cache(maxsize=None)
def local_center(kind: str) -> tuple[float, float, float]:
    """The centre of a part's bounding box in its own frame."""
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
    """One of a task's parts as a task-file object: ``pos`` is the part's origin, ``size`` its bounding box,
    ``kind`` names its entry in ``CUSTOM_KINDS``. Spawned by :func:`arena_object` once :func:`install` has run."""

    kind: str = ""

    def _check(self) -> None:
        super()._check()
        if self.kind not in CUSTOM_KINDS:
            raise ValueError(f"{self.name}: unknown part kind {self.kind!r}")


# -- USD (after the simulator has started) ------------------------------------------------------------------

_USD_DIR: Path | None = None
# kind -> (write_usd(path), arena_object(spec), state(base, name)); a kind may also give step(base, name, states),
# run once per control step before its state is read.
CUSTOM_KINDS: dict[str, tuple] = {}
_CUSTOM: dict[str, str] = {}  # parts spawned so far: name -> kind


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


def usd_file(kind: str) -> str:
    global _USD_DIR
    if _USD_DIR is None:
        _USD_DIR = Path(tempfile.mkdtemp(prefix="task_parts_"))
    path = _USD_DIR / f"{kind}.usda"
    if not path.exists():
        CUSTOM_KINDS[kind][0](path)
    return str(path)


def arena_object(spec: Part):
    """The Arena object for a part, made by its kind."""
    _CUSTOM[spec.name] = spec.kind
    return CUSTOM_KINDS[spec.kind][1](spec)


# -- the hook into yam_arena.scene --------------------------------------------------------------------------


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
        base = env.unwrapped
        custom = {n for n in names if n in _CUSTOM}
        out = original_states(env, [n for n in names if n not in custom])
        for n in sorted(custom):
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
