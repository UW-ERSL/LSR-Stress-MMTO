"""
shapes.py
=========
A part described as a list of steps, so it can be made in LSR MMTO instead of imported as an STL. The
functions here only make plain data (dicts and lists of numbers): the same data a project file keeps under
part_shapes, so a part written in a script is the part saved with the project. part_from_shapes.py builds
it into the STL a run meshes. Lengths are in the project's length unit, angles in degrees.

    from lsr_mmto import Project, shapes as S
    part = [S.add(S.extrusion(S.polygon([(0, 0), (100, 0), (100, 30), (30, 30), (30, 100), (0, 100)]), 10)),
            S.cut(S.cylinder(radius=6, height=10), at=(80, 15, 0)),
            S.cut(S.cylinder(radius=6, height=10), at=(15, 80, 0))]
    p = Project("my L-bracket", part_shapes=part)

A step adds, cuts or keeps the overlap of one shape with the part made so far; the first step adds. Every
shape, outline, repeat and step, with what it means and example parts, is in docs/part_shapes.md.
"""
import csv
import json
import numbers
from typing import Any

Number = int | float
Step = dict[str, Any]

ACTIONS = ("add", "cut", "overlap")
SHAPES = ("box", "cylinder", "sphere", "extrusion", "revolution", "wrap")
OUTLINES = ("rectangle", "circle", "polygon")
AXES = ("x", "y", "z")


class ShapeError(ValueError):
    """A part that cannot be made, with the step it is about in its message."""


# ---------------------------------------------------------------------- shapes
def box(x: Number, y: Number, z: Number) -> dict[str, Any]:
    """A box x by y by z; its origin is its lowest corner."""
    return {"kind": "box", "size": [float(x), float(y), float(z)]}


def cylinder(radius: Number, height: Number, top_radius: Number | None = None) -> dict[str, Any]:
    """A cylinder along z, its origin the centre of its base; a top_radius other than radius makes a cone
    (0 for a point)."""
    return {"kind": "cylinder", "radius": float(radius), "height": float(height),
            "top_radius": None if top_radius is None else float(top_radius)}


def sphere(radius: Number) -> dict[str, Any]:
    """A sphere, its origin its centre."""
    return {"kind": "sphere", "radius": float(radius)}


def extrusion(outline: dict[str, Any], height: Number) -> dict[str, Any]:
    """An outline (rectangle, circle, polygon) drawn in the x-y plane and pushed up z by height."""
    return {"kind": "extrusion", "outline": outline, "height": float(height)}


def revolution(outline: dict[str, Any], angle: Number = 360) -> dict[str, Any]:
    """An outline drawn as (radius, height) spun around z by angle degrees: a ring, a flange, a shaft."""
    return {"kind": "revolution", "outline": outline, "angle": float(angle)}


def place(shape: dict[str, Any], at: tuple = (0, 0, 0), rotate: tuple = (0, 0, 0)) -> dict[str, Any]:
    """A shape turned by rotate and moved to at, as one member of a wrap."""
    return {"shape": shape, "at": _three(at), "rotate": _three(rotate)}


def wrap(*placed: dict[str, Any]) -> dict[str, Any]:
    """The tightest convex shape around two or more placed shapes (place): two cylinders make a lug."""
    return {"kind": "wrap", "shapes": list(placed)}


# ---------------------------------------------------------------------- outlines
def rectangle(width: Number, depth: Number, round: Number = 0) -> dict[str, Any]:
    """A rectangle from (0, 0) to (width, depth); round rounds its corners by that radius, at most half the
    narrower side (half makes a slot)."""
    return {"kind": "rectangle", "width": float(width), "depth": float(depth), "round": float(round)}


def circle(radius: Number) -> dict[str, Any]:
    """A circle centred on (0, 0)."""
    return {"kind": "circle", "radius": float(radius), "round": 0.0}


def polygon(points: list, round: Number = 0) -> dict[str, Any]:
    """The outline through points, (x, y) pairs in order (either way round); round rounds every corner, and is
    less than half the narrowest width of the outline."""
    return {"kind": "polygon", "points": [[float(x), float(y)] for x, y in points], "round": float(round)}


def points_from_csv(path: str) -> list[list[float]]:
    """The (x, y) points in the first two columns of a CSV file, in order; rows that do not start with two
    numbers (a header) are skipped."""
    points = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            try:
                points.append([float(row[0]), float(row[1])])
            except (ValueError, IndexError):
                continue
    return points


# ---------------------------------------------------------------------- repeats
def along_line(count: int, step: tuple) -> dict[str, Any]:
    """count copies, each moved by step (x, y, z) from the one before; the first is where at puts it."""
    return {"kind": "along_line", "count": count, "step": _three(step)}


def around_circle(count: int, centre: tuple = (0, 0, 0), axis: str = "z") -> dict[str, Any]:
    """count copies spread evenly over a full turn about the line through centre along axis ("x", "y" or
    "z"); the first is where at puts it."""
    return {"kind": "around_circle", "count": count, "centre": _three(centre), "axis": axis}


# ---------------------------------------------------------------------- steps
def add(shape: dict[str, Any], at: tuple = (0, 0, 0), rotate: tuple = (0, 0, 0),
        repeat: dict[str, Any] | None = None) -> Step:
    """A step that adds the shape to the part."""
    return _step("add", shape, at, rotate, repeat)


def cut(shape: dict[str, Any], at: tuple = (0, 0, 0), rotate: tuple = (0, 0, 0),
        repeat: dict[str, Any] | None = None) -> Step:
    """A step that cuts the shape out of the part."""
    return _step("cut", shape, at, rotate, repeat)


def overlap(shape: dict[str, Any], at: tuple = (0, 0, 0), rotate: tuple = (0, 0, 0),
            repeat: dict[str, Any] | None = None) -> Step:
    """A step that keeps only the part inside the shape."""
    return _step("overlap", shape, at, rotate, repeat)


def _step(action: str, shape: dict[str, Any], at: tuple, rotate: tuple, repeat: dict[str, Any] | None) -> Step:
    return {"action": action, "shape": shape, "at": _three(at), "rotate": _three(rotate), "repeat": repeat}


def _three(values: tuple) -> list[float]:
    x, y, z = values
    return [float(x), float(y), float(z)]


# ---------------------------------------------------------------------- checking
def check(steps: list[Step]) -> None:
    """ShapeError naming the step, for anything that cannot be built: an unknown action, shape, outline or
    repeat, a size of zero or less, a polygon of fewer than 3 points, a wrap of fewer than 2 shapes, or a first
    step that does not add. What only building can find (a step that leaves nothing) part_from_shapes finds."""
    if not isinstance(steps, list):
        raise ShapeError("a part's shapes are a list of steps: [S.add(...), S.cut(...), ...]")
    if not steps:
        raise ShapeError("a part needs at least one step, and the first one adds a shape")
    for n, step in enumerate(steps, 1):
        where = f"step {n}"
        if not isinstance(step, dict) or step.get("action") not in ACTIONS:
            raise ShapeError(f"{where}: the action is add, cut or overlap")
        if n == 1 and step["action"] != "add":
            raise ShapeError("step 1: the first step adds a shape; there is nothing yet to cut or overlap")
        _check_placement(step, where)
        _check_shape(step.get("shape"), where)
        repeat = step.get("repeat")
        if repeat is not None:
            _check_repeat(repeat, where)


def _check_placement(placed: dict[str, Any], where: str) -> None:
    for key in ("at", "rotate"):
        value = placed.get(key)
        if not _numbers(value, 3):
            raise ShapeError(f"{where}: {key} must be three numbers (x, y, z)")


def _check_shape(shape: Any, where: str) -> None:
    if not isinstance(shape, dict) or shape.get("kind") not in SHAPES:
        raise ShapeError(f"{where}: the shape is one of {', '.join(SHAPES)}")
    kind = shape["kind"]
    if kind == "box":
        size = shape.get("size")
        if not _numbers(size, 3):
            raise ShapeError(f"{where}: a box's size must be three numbers (x, y, z), in a list")
        if not all(_positive(v) for v in size):
            raise ShapeError(f"{where}: a box's three sizes must be above 0")
    elif kind == "cylinder":
        _positive_fields(shape, ("radius", "height"), where, "cylinder")
        top = shape.get("top_radius")
        if top is not None and not (_number(top) and top >= 0):
            raise ShapeError(f"{where}: a cylinder's top radius is 0 or more, or left out")
    elif kind == "sphere":
        _positive_fields(shape, ("radius",), where, "sphere")
    elif kind in ("extrusion", "revolution"):
        _check_outline(shape.get("outline"), where)
        if kind == "extrusion":
            _positive_fields(shape, ("height",), where, "extrusion")
        elif not (_number(shape.get("angle")) and 0 < shape["angle"] <= 360):
            raise ShapeError(f"{where}: a revolution's angle must be above 0 and at most 360 degrees")
    else:                                                   # wrap
        members = shape.get("shapes")
        if not isinstance(members, list) or len(members) < 2:
            raise ShapeError(f"{where}: a wrap is around two or more placed shapes")
        for k, member in enumerate(members, 1):
            if not isinstance(member, dict) or "shape" not in member:
                raise ShapeError(f"{where}, wrapped shape {k}: each wrapped shape is given as "
                                 f"S.place(shape, at=..., rotate=...)")
            _check_placement(member, f"{where}, wrapped shape {k}")
            _check_shape(member.get("shape"), f"{where}, wrapped shape {k}")


def _check_outline(outline: Any, where: str) -> None:
    if not isinstance(outline, dict) or outline.get("kind") not in OUTLINES:
        raise ShapeError(f"{where}: the outline is one of {', '.join(OUTLINES)}")
    kind = outline["kind"]
    if kind == "rectangle":
        _positive_fields(outline, ("width", "depth"), where, "rectangle")
        if _number(outline.get("round", 0)) and _number(outline["width"]) and _number(outline["depth"]) and \
                outline.get("round", 0) > min(outline["width"], outline["depth"]) / 2:
            raise ShapeError(f"{where}: a rectangle's corners are rounded by at most half its narrower side "
                             f"({min(outline['width'], outline['depth']) / 2:g})")
    elif kind == "circle":
        _positive_fields(outline, ("radius",), where, "circle")
        if outline.get("round", 0):
            raise ShapeError(f"{where}: a circle has no corners to round")
    else:
        points = outline.get("points")
        if not (isinstance(points, (list, tuple)) and len(points) >= 3 and all(_numbers(p, 2) for p in points)):
            raise ShapeError(f"{where}: a polygon is three or more (x, y) points")
    if not (_number(outline.get("round", 0)) and outline.get("round", 0) >= 0):
        raise ShapeError(f"{where}: an outline's corner rounding is 0 or more")


def _check_repeat(repeat: Any, where: str) -> None:
    if not isinstance(repeat, dict) or repeat.get("kind") not in ("along_line", "around_circle"):
        raise ShapeError(f"{where}: a repeat is along_line or around_circle")
    count = repeat.get("count")
    if isinstance(count, bool) or not isinstance(count, numbers.Integral) or count < 1:
        raise ShapeError(f"{where}: a repeat's count must be a whole number, 1 or more")
    key = "step" if repeat["kind"] == "along_line" else "centre"
    value = repeat.get(key)
    if not _numbers(value, 3):
        raise ShapeError(f"{where}: a repeat's {key} must be three numbers (x, y, z)")
    if repeat["kind"] == "around_circle" and repeat.get("axis") not in AXES:
        raise ShapeError(f"{where}: a repeat around a circle turns about the x, y or z axis")


def _positive_fields(shape: dict[str, Any], keys: tuple, where: str, what: str) -> None:
    for key in keys:
        if not _positive(shape.get(key)):
            article = "an" if what[0] in "aeiou" else "a"
            raise ShapeError(f"{where}: {article} {what}'s {key.replace('_', ' ')} must be above 0")


def _number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and value == value \
        and abs(value) != float("inf")


def _numbers(values: Any, count: int) -> bool:
    return isinstance(values, (list, tuple)) and len(values) == count and all(_number(v) for v in values)



def plain(steps: list[Step]) -> list[Step]:
    """The steps as a project file keeps them: made again with the functions above, so every key is there
    (round 0, no repeat, ...) and every number is a float but a repeat's count. Two lists of the same steps
    are then equal however they were written. Steps that check refuses are only made JSON data (lists for
    tuples, int and float for NumPy numbers), so they can still be saved and reported."""
    try:
        check(steps)
    except ShapeError:
        def number(value):
            if isinstance(value, numbers.Integral):
                return int(value)
            if isinstance(value, numbers.Real):
                return float(value)
            raise TypeError(f"{value!r} is not a number")
        return json.loads(json.dumps(steps, default=number))
    return [_remade_step(step) for step in steps]


def _remade_step(step: Step) -> Step:
    r = step.get("repeat")
    if r is not None:
        r = along_line(int(r["count"]), r["step"]) if r["kind"] == "along_line" else \
            around_circle(int(r["count"]), r["centre"], r["axis"])
    return _step(step["action"], _remade_shape(step["shape"]), step["at"], step["rotate"], r)


def _remade_shape(s: dict[str, Any]) -> dict[str, Any]:
    kind = s["kind"]
    if kind == "box":
        return box(*s["size"])
    if kind == "cylinder":
        return cylinder(s["radius"], s["height"], s.get("top_radius"))
    if kind == "sphere":
        return sphere(s["radius"])
    if kind == "extrusion":
        return extrusion(_remade_outline(s["outline"]), s["height"])
    if kind == "revolution":
        return revolution(_remade_outline(s["outline"]), s["angle"])
    return wrap(*[place(_remade_shape(m["shape"]), m["at"], m["rotate"]) for m in s["shapes"]])


def _remade_outline(o: dict[str, Any]) -> dict[str, Any]:
    if o["kind"] == "rectangle":
        return rectangle(o["width"], o["depth"], o.get("round", 0))
    if o["kind"] == "circle":
        return circle(o["radius"])
    return polygon(o["points"], o.get("round", 0))


def _positive(value: Any) -> bool:
    return _number(value) and value > 0


# ---------------------------------------------------------------------- whole parts
def scaled(steps: list[Step], factor: float) -> list[Step]:
    """The steps with every length (sizes, radii, heights, corner points, rounding, positions, repeat steps
    and centres) times factor; angles and counts stay. Scale... in Geometry scales a part made of shapes so."""
    out = plain(steps)

    def placement(placed):
        placed["at"] = [float(v * factor) for v in placed["at"]]

    def outline(o):
        for key in ("width", "depth", "radius", "round"):
            if key in o:
                o[key] = float(o[key] * factor)
        if "points" in o:
            o["points"] = [[float(x * factor), float(y * factor)] for x, y in o["points"]]

    def shape(s):
        for key in ("radius", "height"):
            if key in s:
                s[key] = float(s[key] * factor)
        if s.get("top_radius") is not None:
            s["top_radius"] = float(s["top_radius"] * factor)
        if "size" in s:
            s["size"] = [float(v * factor) for v in s["size"]]
        if "outline" in s:
            outline(s["outline"])
        for member in s.get("shapes", ()):
            placement(member)
            shape(member["shape"])

    for step in out:
        placement(step)
        shape(step["shape"])
        repeat = step.get("repeat")
        if repeat is not None:
            for key in ("step", "centre"):
                if key in repeat:
                    repeat[key] = [float(v * factor) for v in repeat[key]]
    return out


def describe(step: Step) -> str:
    """One line for a step, as a list of steps shows it: "cut cylinder r 6, height 10 at (80, 15, 0)"."""
    s = step["shape"]
    kind = s["kind"]
    if kind == "box":
        what = "box " + " x ".join(_short(v) for v in s["size"])
    elif kind == "cylinder":
        what = f"{'cone' if s.get('top_radius') not in (None, s['radius']) else 'cylinder'} r {_short(s['radius'])}, " \
               f"height {_short(s['height'])}"
    elif kind == "sphere":
        what = f"sphere r {_short(s['radius'])}"
    elif kind == "extrusion":
        what = f"extrusion of a {s['outline']['kind']}, height {_short(s['height'])}"
    elif kind == "revolution":
        what = f"revolution of a {s['outline']['kind']} spun {_short(s['angle'])} degrees"
    else:
        what = f"wrap of {len(s['shapes'])} shapes"
    line = f"{step['action']} {what} at ({', '.join(_short(v) for v in step['at'])})"
    repeat = step.get("repeat")
    if repeat is not None and repeat["count"] > 1:
        line += f", {repeat['count']} times " + ("along a line" if repeat["kind"] == "along_line" else "around a circle")
    return line


def _short(value: float) -> str:
    return f"{value:g}"


def to_script(steps: list[Step], indent: str = "    ") -> list[str]:
    """The lines of a Python list that makes the steps with the calls above (S. before each), as a project
    script writes it: S.add(S.box(100, 40, 10), at=(0, 0, 0)), ..."""
    return [f"{indent}{_step_call(step)}," for step in plain(steps)]


def _step_call(step: Step) -> str:
    args = [_shape_call(step["shape"])] + _placement_args(step)
    if step.get("repeat") is not None:
        args.append(f"repeat={_repeat_call(step['repeat'])}")
    return f"S.{step['action']}({', '.join(args)})"


def _placement_args(placed: dict[str, Any]) -> list[str]:
    args = []
    if any(placed["at"]):
        args.append(f"at={_tuple(placed['at'])}")
    if any(placed["rotate"]):
        args.append(f"rotate={_tuple(placed['rotate'])}")
    return args


def _shape_call(s: dict[str, Any]) -> str:
    kind = s["kind"]
    if kind == "box":
        return f"S.box({', '.join(map(_r, s['size']))})"
    if kind == "cylinder":
        top = "" if s.get("top_radius") is None else f", top_radius={_r(s['top_radius'])}"
        return f"S.cylinder(radius={_r(s['radius'])}, height={_r(s['height'])}{top})"
    if kind == "sphere":
        return f"S.sphere(radius={_r(s['radius'])})"
    if kind == "extrusion":
        return f"S.extrusion({_outline_call(s['outline'])}, height={_r(s['height'])})"
    if kind == "revolution":
        return f"S.revolution({_outline_call(s['outline'])}, angle={_r(s['angle'])})"
    members = [f"S.place({', '.join([_shape_call(m['shape'])] + _placement_args(m))})" for m in s["shapes"]]
    return f"S.wrap({', '.join(members)})"


def _outline_call(o: dict[str, Any]) -> str:
    round_ = f", round={_r(o['round'])}" if o.get("round") else ""
    if o["kind"] == "rectangle":
        return f"S.rectangle({_r(o['width'])}, {_r(o['depth'])}{round_})"
    if o["kind"] == "circle":
        return f"S.circle({_r(o['radius'])})"
    points = ", ".join(f"({_r(x)}, {_r(y)})" for x, y in o["points"])
    return f"S.polygon([{points}]{round_})"


def _repeat_call(r: dict[str, Any]) -> str:
    if r["kind"] == "along_line":
        return f"S.along_line({r['count']}, step={_tuple(r['step'])})"
    return f"S.around_circle({r['count']}, centre={_tuple(r['centre'])}, axis={r['axis']!r})"


def _tuple(values: list[float]) -> str:
    return f"({', '.join(_r(v) for v in values)})"


def _r(value: float) -> str:
    return repr(float(value))
