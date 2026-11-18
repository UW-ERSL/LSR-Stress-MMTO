"""
part_from_shapes.py
===================
Builds a part made of shapes (shapes.py) into the closed STL a run meshes, the way an imported STL is
used: the mesh, the selections, the supports and loads see no difference.

    from lsr_mmto.problem.part_from_shapes import part_stl
    path = part_stl(p)          # p's part as an STL, built only when no STL of these shapes is there yet

The STL is part/shapes_<fingerprint>.stl in the project's folder. The fingerprint is the first 6 characters
of the SHA-256 of the steps, so the same steps always give the same file and changed steps a new one: a run's
record and the project as it is now never share a file once the shapes differ. When a project builds its part,
the parts built before in that folder are removed; a run's project (Project.from_record) builds its own part
again from the steps its record keeps, and removes nothing, so showing an old run never removes the part the
project is using. One project made of shapes to a folder, as the GUI makes them.

Curves are made of flat pieces that stay within TOLERANCE_OF_PART_SIZE of the part's size (its diagonal) of
the true curve, far smaller than an element. The pieces depend on the steps alone, never on the mesh settings,
so the STL's triangles, and the supports and loads picked on them, stay the same when Mesh dofs changes.

Steps of the same kind in a row (adds, or cuts) are joined in one pass of the geometry library (manifold3d),
since cutting one shape at a time works on the whole part each time. A step that leaves nothing, a polygon
whose edges cross, and rounding that removes a narrow part of an outline are refused (ShapeError, naming the
step). A cut that misses the part and a part in pieces are warned about when it is built; a feature under 2
elements across, at the project's mesh settings, is warned about by small_feature_warnings.
"""
import hashlib
import json
import math
import os
import re
import tempfile
from typing import TYPE_CHECKING, Any

import numpy as np

from lsr_mmto.problem.shapes import ShapeError, Step, check, describe, plain

if TYPE_CHECKING:
    from lsr_mmto.problem.project import Project

BUILD_VERSION = 1                   # in the fingerprint: a change to how parts are built builds every part again
ROUGH_PIECES = 32                   # pieces per circle in the first pass, which only finds the part's size
PIECES_PER_CIRCLE = (16, 1024)      # the fewest and the most pieces per circle
TOLERANCE_OF_PART_SIZE = 1e-4       # how far a curve may stand from its pieces, as a share of the part's diagonal
SMALLEST_FEATURE = 2.0              # a feature fewer elements across than this may vanish in the mesh
PREFIX = "shapes_"                  # the built files: shapes_<fingerprint>.stl, and .json beside it
BUILT = re.compile(rf"{PREFIX}[0-9a-f]{{6}}\.(stl|json)")     # a built file's name; no other file is removed


def part_stl(project: "Project") -> str:
    """The path of the project's part built into an STL (stl_name, in part/ of its folder), built when it is
    not there yet. A project's own build removes the parts built before in that folder; a run's project
    (from_record) removes nothing. ShapeError for a part that cannot be made; OSError when the folder cannot be
    written."""
    path = stl_path(project)
    facts = _built_facts(path) if os.path.isfile(path) else None
    if facts is None or facts.get("fingerprint") != _fingerprint(project.part_shapes):
        surface, facts = build_surface(project.part_shapes)
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        _write(path, lambda written: surface.export(written, file_type="stl"))
        _write(_facts_path(path), lambda written: _dump(facts, written))
        if not getattr(project, "_from_record", False):
            _remove_older(folder, os.path.basename(path))
    return path


def stl_path(project: "Project") -> str:
    """Where the project's part made of shapes is built: part/shapes_<fingerprint>.stl in its folder, built or
    not."""
    from lsr_mmto.problem.project_folder import PART
    return os.path.join(project.base_dir, PART, stl_name(project.part_shapes))


def stl_name(steps: list[Step]) -> str:
    """shapes_<fingerprint>.stl: the fingerprint is the first 6 characters of the SHA-256 of the steps (and of
    BUILD_VERSION), so the same steps always give the same name. ShapeError for steps that cannot be built."""
    return f"{PREFIX}{_fingerprint(steps)[:6]}.stl"


def build_warnings(project: "Project") -> list[str]:
    """What building the project's part found: a cut that misses the part, a part in pieces. ShapeError for a
    part that cannot be made."""
    return list(_built_facts(part_stl(project))["warnings"])


def small_feature_warnings(project: "Project") -> list[str]:
    """A warning for each step whose shape is under SMALLEST_FEATURE elements across at the project's mesh
    settings, with the Mesh dofs that would show it. ShapeError for a part that cannot be made."""
    facts = _built_facts(part_stl(project))
    h = element_size(facts["bounds"], facts["volume"], project.n_dof, project.single_layer,
                     project.unknowns_per_node())
    return _small_features(project.part_shapes, h, project.n_dof, project.single_layer)


def build_surface(steps: list[Step]) -> tuple[Any, dict[str, Any]]:
    """(surface, facts): the steps built into a closed triangle surface (a trimesh.Trimesh), and what the build
    found: {"fingerprint", "warnings", "bounds" (x, y, z lowest then highest), "volume"}. Built once with
    ROUGH_PIECES pieces per circle to find the part's size, then again with pieces fine for that size."""
    import trimesh
    check(steps)
    rough, _ = _apply_steps(steps, None)
    bounds = np.asarray(rough.bounding_box(), dtype=float)
    tolerance = float(np.linalg.norm(bounds[3:] - bounds[:3])) * TOLERANCE_OF_PART_SIZE
    part, warnings = _apply_steps(steps, tolerance)
    pieces = len(part.decompose())
    if pieces > 1:
        warnings.append(f"the part is in {pieces} separate pieces; a run treats them as one part")
    mesh = part.to_mesh()
    surface = trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=True)
    facts = {"fingerprint": _fingerprint(steps), "warnings": warnings,
             "bounds": [float(v) for v in part.bounding_box()], "volume": float(part.volume())}
    return surface, facts


def element_size(bounds: list[float], volume: float, n_dof: int, single_layer: bool,
                 unknowns_per_node: int) -> np.ndarray:
    """The element size (x, y, z) the mesher chooses for a part of these bounds (x, y, z lowest then highest)
    and volume, as PyTO-AD's hex_mesher works it out: cubes of a size that fits 0.9 x n_dof / unknowns_per_node
    of them in the part's volume, then as many along each side as come nearest to that size, at least 2 (at
    least 1 for a single layer)."""
    bounds = np.asarray(bounds, dtype=float)
    sides = bounds[3:] - bounds[:3]
    alpha = (0.9 * (n_dof / unknowns_per_node) / volume) ** (1 / 3)
    counts = np.maximum(np.round(alpha * sides), 1 if single_layer else 2)
    return sides / counts


# ---------------------------------------------------------------------- the built files
def _fingerprint(steps: list[Step]) -> str:
    check(steps)
    data = {"steps": plain(steps), "build_version": BUILD_VERSION}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _facts_path(stl: str) -> str:
    return os.path.splitext(stl)[0] + ".json"


def _built_facts(stl: str) -> dict[str, Any] | None:
    """The facts written beside a built STL, or None when they are missing or unreadable (the part is then
    built again)."""
    try:
        with open(_facts_path(stl)) as fh:
            facts = json.load(fh)
        return facts if {"warnings", "bounds", "volume"} <= set(facts) else None
    except (OSError, ValueError):
        return None


def _dump(facts: dict[str, Any], path: str) -> None:
    with open(path, "w") as fh:
        json.dump(facts, fh, indent=1)


def _write(path: str, write) -> None:
    """write(a path) writes the file in full beside path, which then takes its place at once, so a reader of
    path never sees half a file."""
    handle, written = tempfile.mkstemp(dir=os.path.dirname(path), prefix=os.path.basename(path) + ".",
                                       suffix=".writing")           # a name of its own for each writer
    os.close(handle)
    try:
        write(written)
        os.replace(written, path)
    finally:
        if os.path.exists(written):
            os.remove(written)


def _remove_older(folder: str, keep: str) -> None:
    """Every built part in folder but keep (and the facts beside each) removed; only names of BUILT, so a file
    of the user's own is never touched; one open elsewhere is left."""
    stem = os.path.splitext(keep)[0]
    for name in os.listdir(folder):
        if BUILT.fullmatch(name) and os.path.splitext(name)[0] != stem:
            try:
                os.remove(os.path.join(folder, name))
            except OSError:
                pass


# ---------------------------------------------------------------------- building
def _apply_steps(steps: list[Step], tolerance: float | None) -> tuple[Any, list[str]]:
    """(the part, warnings): the steps applied in order, adds and cuts in a row joined into one pass. A cut step
    whose shape (or some of its copies) misses the part is warned about."""
    import manifold3d as m
    part, warnings = None, []
    n = 0
    while n < len(steps):
        action = steps[n]["action"]
        last = n
        if action in ("add", "cut"):
            while last + 1 < len(steps) and steps[last + 1]["action"] == action:
                last += 1
        copies = {k: _placed_copies(steps[k], k + 1, tolerance) for k in range(n, last + 1)}
        group = m.Manifold.batch_boolean([c for k in copies for c in copies[k]], m.OpType.Add)
        if action == "cut":
            warnings += [w for w in (_missed(k + 1, part, copies[k]) for k in copies) if w]
        if part is None:
            part = group
        elif action == "add":
            part = part + group
        elif action == "cut":
            part = part - group
        else:
            part = part ^ group
        if part.is_empty() or part.volume() <= 0:
            which = f"step {n + 1}" if last == n else f"steps {n + 1} to {last + 1}"
            raise ShapeError(f"{which} leave{'s' if last == n else ''} nothing of the part")
        n = last + 1
    return part, warnings


def _missed(number: int, part: Any, copies: list[Any]) -> str | None:
    """The warning for a cut step whose copies, some or all, do not touch the part as it was before the cut."""
    missing = sum(1 for c in copies if (part ^ c).is_empty() or (part ^ c).volume() <= 0)
    if missing == 0:
        return None
    if missing == len(copies):
        return f"step {number} cuts nothing: it misses the part"
    return f"step {number}: {missing} of its {len(copies)} copies cut nothing: they miss the part"


def _placed_copies(step: Step, number: int, tolerance: float | None) -> list[Any]:
    """One step's shape, turned, moved and repeated: one solid per copy."""
    shape = _placed(_shape(step["shape"], f"step {number}", tolerance), step["at"], step["rotate"])
    repeat = step.get("repeat")
    if repeat is None or repeat["count"] == 1:
        return [shape]
    if repeat["kind"] == "along_line":
        d = np.asarray(repeat["step"], dtype=float)
        return [shape.translate(list(d * i)) for i in range(int(repeat["count"]))]
    c = np.asarray(repeat["centre"], dtype=float)
    axis = np.eye(3)[("x", "y", "z").index(repeat["axis"])]
    turn = 360.0 / int(repeat["count"])
    return [shape.translate(list(-c)).rotate(list(axis * turn * i)).translate(list(c))
            for i in range(int(repeat["count"]))]


def _placed(shape: Any, at: list[float], rotate: list[float]) -> Any:
    return shape.rotate([float(v) for v in rotate]).translate([float(v) for v in at])


def _shape(s: dict[str, Any], where: str, tolerance: float | None) -> Any:
    import manifold3d as m
    kind = s["kind"]
    if kind == "box":
        return m.Manifold.cube([float(v) for v in s["size"]])
    if kind == "cylinder":
        top = s["radius"] if s.get("top_radius") is None else s["top_radius"]
        return m.Manifold.cylinder(float(s["height"]), float(s["radius"]), float(top),
                                   _pieces(max(s["radius"], top), tolerance))
    if kind == "sphere":
        return m.Manifold.sphere(float(s["radius"]), _pieces(s["radius"], tolerance))
    if kind == "extrusion":
        return m.Manifold.extrude(_outline(s["outline"], where, tolerance), float(s["height"]))
    if kind == "revolution":
        if s["outline"]["kind"] == "circle":
            raise ShapeError(f"{where}: a circle is centred on the axis it would spin around; spin a polygon "
                             f"or a rectangle drawn at x of 0 or more")
        outline = _outline(s["outline"], where, tolerance)
        lo, hi = outline.bounds()[:2], outline.bounds()[2:]
        if lo[0] < 0:
            raise ShapeError(f"{where}: an outline spun around z is drawn at x of 0 or more (x is the radius)")
        return m.Manifold.revolve(outline, _pieces(hi[0], tolerance), float(s["angle"]))
    return m.Manifold.batch_hull([_placed(_shape(p["shape"], where, tolerance), p["at"], p["rotate"])
                                  for p in s["shapes"]])


def _outline(o: dict[str, Any], where: str, tolerance: float | None) -> Any:
    import manifold3d as m
    r = float(o.get("round") or 0)
    if o["kind"] == "rectangle":
        w, d = float(o["width"]), float(o["depth"])
        if r <= 0:
            return m.CrossSection.square([w, d])
        n = _pieces(r, tolerance)                            # four rounded corners: the hull of four circles
        return m.CrossSection.batch_hull([m.CrossSection.circle(r, n).translate([x, y])
                                          for x, y in ((r, r), (w - r, r), (w - r, d - r), (r, d - r))])
    if o["kind"] == "circle":
        return m.CrossSection.circle(float(o["radius"]), _pieces(o["radius"], tolerance))
    points = np.asarray(o["points"], dtype=float)
    x, y = points[:, 0], points[:, 1]
    area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if area < 0:                                             # either way round: made anticlockwise
        points = points[::-1]
    outline = m.CrossSection([points.tolist()])
    if abs(area) == 0 or not math.isclose(outline.area(), abs(area), rel_tol=1e-6):
        raise ShapeError(f"{where}: the polygon's edges cross or it has no area; give its corner points "
                         f"in order around its edge")
    if r <= 0:
        return outline
    n = _pieces(r, tolerance)
    rounded = outline.offset(r, m.JoinType.Round, 2.0, n).offset(-2 * r, m.JoinType.Round, 2.0, n) \
                     .offset(r, m.JoinType.Round, 2.0, n)
    expected, slack = _rounding_change(points, r)
    if rounded.is_empty() or len(rounded.decompose()) != len(outline.decompose()) or \
            abs(rounded.area() - (abs(area) + expected)) > slack + 1e-9 * abs(area):
        raise ShapeError(f"{where}: rounding by {r:g} does more than round the corners: a part of the polygon "
                         f"narrower than {2 * r:g} (twice the rounding) is lost or filled in, or an edge is too "
                         f"short for the rounding at its ends; round by less")
    return rounded


def _rounding_change(points: np.ndarray, r: float) -> tuple[float, float]:
    """(change, slack): how much rounding every corner of the polygon (points anticlockwise) by r changes its
    area when the roundings only touch their own corners, and how far the flat pieces of the arcs may move
    that. A corner whose edges meet at angle a (its sharp side) loses, or for a corner pointing inwards gains,
    r^2 (1 / tan(a/2) - (pi - a)/2): the corner's sharp tip outside the arc."""
    change = slack = 0.0
    for i in range(len(points)):
        before, here, after = points[i - 1], points[i], points[(i + 1) % len(points)]
        a, b = before - here, after - here
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            continue
        angle = math.acos(max(-1.0, min(1.0, float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))))
        if angle <= 0:
            continue
        tip = r * r * (1 / math.tan(angle / 2) - (math.pi - angle) / 2)
        inward = (here[0] - before[0]) * (after[1] - here[1]) - (here[1] - before[1]) * (after[0] - here[0]) < 0
        change += tip if inward else -tip
        slack += 0.05 * (math.pi - angle) * r * r
    return change, slack


def _pieces(radius: float, tolerance: float | None) -> int:
    """Pieces per full circle of this radius so that no piece stands further than tolerance from it, within
    PIECES_PER_CIRCLE; ROUGH_PIECES before the part's size is known (tolerance None)."""
    if tolerance is None:
        return ROUGH_PIECES
    if tolerance >= radius:
        return PIECES_PER_CIRCLE[0]
    n = math.ceil(math.pi / math.acos(1 - tolerance / radius))
    return int(min(max(n, PIECES_PER_CIRCLE[0]), PIECES_PER_CIRCLE[1]))


# ---------------------------------------------------------------------- small features
def _small_features(steps: list[Step], h: np.ndarray, n_dof: int, single_layer: bool) -> list[str]:
    """A warning for each step whose shape is fewer than SMALLEST_FEATURE elements across, with the Mesh dofs
    that would give it that many. A single layer is one element thick by choice, so there only the sizes
    across the plate count (a shape turned by its step counts every size)."""
    out = []
    element = float(h[:2].min()) if single_layer else float(h.min())
    for n, step in enumerate(steps, 1):
        flat = single_layer and not any(step["rotate"])
        size = _smallest_size(step["shape"], flat)
        if size is None or size >= SMALLEST_FEATURE * element:
            continue
        finer = (SMALLEST_FEATURE * element / size) ** (2 if single_layer else 3)
        needed = int(math.ceil(n_dof * finer / 1000.0)) * 1000
        out.append(f"step {n} ({describe(step)}) is {size:g} across at its narrowest, under {SMALLEST_FEATURE:g} "
                   f"elements of {element:.3g}, so it may vanish or come out square in the mesh; Mesh dofs of "
                   f"about {needed} would show it")
    return out


def _smallest_size(s: dict[str, Any], across_only: bool) -> float | None:
    """The shape's narrowest size; with across_only, its size along z (a height) is left out."""
    kind = s["kind"]
    if kind == "box":
        return min(s["size"][:2] if across_only else s["size"])
    if kind == "cylinder":
        return 2 * s["radius"] if across_only else min(2 * s["radius"], s["height"])
    if kind == "sphere":
        return 2 * s["radius"]
    if kind == "wrap":
        return None
    o = s["outline"]
    if kind == "revolution" and across_only:             # the outline's x is across the plate, its y the height
        if o["kind"] == "rectangle":
            return o["width"]
        points = np.asarray(o["points"], dtype=float)
        return float(points[:, 0].max() - points[:, 0].min())
    if o["kind"] == "rectangle":
        across = min(o["width"], o["depth"])
    elif o["kind"] == "circle":
        across = 2 * o["radius"]
    else:
        points = np.asarray(o["points"], dtype=float)
        across = float((points.max(axis=0) - points.min(axis=0)).min())
    if kind == "extrusion" and not across_only:
        return min(across, s["height"])
    return across
