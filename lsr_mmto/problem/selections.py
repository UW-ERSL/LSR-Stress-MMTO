"""
selections.py
=============
Named regions of a voxel mesh: which nodes carry a support or a load, which elements
are kept solid. A selection is a description (a box, a cylinder, a face, ...), not a
list of indices, so it holds for any mesh size and can be saved in a project file.

    from lsr_mmto.problem.selections import box, cylinder, face, union
    left = face("xmin")
    bore = cylinder(center=[0.1, 0, 9.5], axis="y", radius=6.0, along=(-3.5, 3.0), pad=0)
    nodes = left.evaluate(mesh, on="nodes")          # sorted node indices
    elems = bore.evaluate(mesh, on="elements")       # sorted element indices (by centers)

on="nodes" tests the mesh nodes, on="elements" the element centers.

pad widens every bound, in model units. The default "auto" is half the smallest element
size for nodes, so a face at x = 0 still catches nodes at x = 1e-12, and 0 for element
centers, which sit half an element from any face and would otherwise pull in the next
layer. plane and face always take half an element by default, since they need a
width: on elements, face gives the outermost layer. plane takes half the element size
along its own axis, so any value inside the part catches the nearest layer of nodes (half
the smallest size, the thickness of a thin plate, leaves gaps between layers). "half"
forces half an element; pad=0 is exact.

Types (to_dict / from_dict give the JSON form a project file stores):
    box(lo, hi)                                  lo <= p <= hi
    cylinder(center, axis, radius, inner_radius=0, along=None)
                                                 distance to the axis in [inner_radius, radius];
                                                 along = (t0, t1) limits (p - center) . axis
    sphere(center, radius)
    plane(axis, value, limits=None)              |p_axis - value| <= pad; limits={"y": [lo, hi]}
                                                 keeps part of the cut
    face(side)                                   the plane at the mesh's extreme, side "xmin" ... "zmax"
    near_face(side, distance)                    within distance of that extreme (strictly)
    nearest(point)                               the single closest node or element
    stl_faces(triangles, tol=None)               boundary nodes on these triangles of the STL (nodes only)
    halfspaces(normals, origins, boundary=True)  inside a set of planes, e.g. a box dragged in the GUI
    nearest(point, boundary=True)                the closest boundary node, as a point picked in the GUI
    rows(face, axis, n, end="max")               the n outermost node rows of a face along an
                                                 axis, e.g. rows("xmax", "y", 8) (nodes only)
    ids(indices)                                 explicit indices; tied to one mesh
    inside_stl(path, shift=(0, 0, 0))            inside a closed STL surface, moved by shift first
    function("file.py:name")                     name(mesh, on) returns the indices
    union(a, b, ...), intersect(a, b, ...), difference(a, b)
    selection("name")                            another selection of the project, by name: a live
                                                 link, looked up when the project evaluates it

A function selection runs the Python file it names when it is evaluated: open project
files with function selections only from people you trust.

Lengths (radius, inner_radius, along, distance, pad) are numbers in model units, or text
with h, the largest element size: "<k>h" for k times it, e.g. near_face("zmin", "1.5h"), or
plain arithmetic in numbers, pi and h, e.g. "5.64 + 1h", "2*h", "38/2 + 0.5h". Text with h
is kept as it is and worked out on each mesh; arithmetic without h is a number at once.

To add a type: subclass Selection, give it a TYPE, a fields tuple and, for each field, how
it is given (field_kinds: a Field each, which the GUI's form is made from), write _mask or
evaluate, and add it to tests/test_selections.py. from_dict finds it by TYPE.
"""
import ast
import importlib.util
import math
import operator
import os
import re

import numpy as np

_AXES = {"x": 0, "y": 1, "z": 2}
_TYPES = {}


def _points(mesh, on):
    if on == "nodes":
        return np.asarray(mesh.node_xyz, dtype=float)
    if on == "elements":
        return np.asarray(mesh.elem_centers, dtype=float)
    raise ValueError(f"on must be 'nodes' or 'elements', not {on!r}")


_ARITHMETIC = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
               ast.Pow: lambda a, b: float(a) ** float(b), ast.USub: operator.neg, ast.UAdd: operator.pos}


def plain_value(node, names=None):
    """The value of a parsed piece of text made only of numbers, text, True/False/None, lists, tuples,
    dicts, and arithmetic on numbers: + - * / ** ( ) and the names in names (pi by default): 38/2,
    2*pi*5. Nothing else is accepted: no other names, no calls, no attributes. ValueError otherwise."""
    names = {"pi": math.pi} if names is None else names
    if isinstance(node, ast.Constant) and not isinstance(node.value, complex):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [plain_value(e, names) for e in node.elts]
        return items if isinstance(node, ast.List) else tuple(items)
    if isinstance(node, ast.Dict) and None not in node.keys:
        return {plain_value(k, names): plain_value(v, names) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    operands = [node.operand] if isinstance(node, ast.UnaryOp) else \
        [node.left, node.right] if isinstance(node, ast.BinOp) else None
    if operands is not None and type(node.op) in _ARITHMETIC:
        values = [plain_value(x, names) for x in operands]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            raise ValueError("arithmetic is only on numbers")
        try:
            return _ARITHMETIC[type(node.op)](*values)
        except (ZeroDivisionError, OverflowError) as e:
            raise ValueError(str(e)) from None
    raise ValueError(f"only numbers and + - * / ** ( ) {' '.join(names)} are accepted")


_K_H = re.compile(r"((?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*h\b")       # "1.5h", "1.5 h": 1.5 times h


def _length_text(v):
    """The parsed form of a length text, "1.5h" read as 1.5 * h; None when it does not parse."""
    try:
        return ast.parse(_K_H.sub(r"(\1*h)", v.strip()), mode="eval").body
    except SyntaxError:
        return None


def _length_value(v, h):
    """The length text v worked out with h (the largest element size), or ValueError."""
    tree = _length_text(v)
    try:
        value = None if tree is None else plain_value(tree, {"pi": math.pi, "h": h})
    except ValueError:
        value = None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"length {v!r}: write a number, or a length in element sizes h, e.g. '1.5h' or "
                         f"'5 + 1h'")
    return value


def _length(mesh, v):
    """A length: a number in model units, or text in numbers, pi and h, the largest element size
    ("1.5h", "5 + 1h"), worked out on this mesh."""
    if isinstance(v, str):
        return float(_length_value(v, float(np.max(mesh.elem_size))))
    return float(v)


def _keep(v):
    """A length as given: a number, or text with h in it kept as text until there is a mesh; text
    without h (38/2) becomes its number."""
    if isinstance(v, str):
        value = float(_length_value(v, 1.0))
        uses_h = any(isinstance(n, ast.Name) and n.id == "h" for n in ast.walk(_length_text(v)))
        return v if uses_h else value
    return float(v)


def _pad(mesh, pad, on, axis=None):
    """pad in model units; "half" is half the smallest element size, or along axis (0, 1, 2) if given,
    or with axis="each" half the size along each axis (three numbers)."""
    if pad == "auto":
        pad = "half" if on == "nodes" else 0.0
    if pad == "half":
        h = np.asarray(mesh.elem_size, dtype=float)
        if axis == "each":
            return 0.5 * h
        return 0.5 * float(np.min(h) if axis is None else h[axis])
    return _length(mesh, pad)


def _axis_vector(axis):
    if isinstance(axis, str):
        v = np.zeros(3)
        v[_AXES[axis]] = 1.0
        return v
    v = np.asarray(axis, dtype=float)
    return v / np.linalg.norm(v)


class Field:
    """How one field of a selection type is given; the GUI's form is made from it. kind is "length"
    (a length, shown beside the project's length unit), "point" (x, y, z, in that unit), "choice"
    (one of choices), "tick" (true or false), "file" (a file; files: the kinds of file offered),
    "number", "numbers" (a list of numbers) or "limits" (the from and to of other directions).
    choices: (shown, value) pairs offered in a list, for any kind; editable: a value of your own may
    be typed too. optional: may be left empty, which means None. help: what the field does, in plain
    words, for the form's (i)."""
    __slots__ = ("kind", "choices", "editable", "optional", "files", "help")

    def __init__(self, kind, choices=(), editable=False, optional=False, files=None, help=None):
        self.kind, self.choices, self.editable = kind, tuple(choices), editable
        self.optional, self.files, self.help = optional, files, help


SIDES = (("Left (x min)", "xmin"), ("Right (x max)", "xmax"), ("Bottom (y min)", "ymin"),
         ("Top (y max)", "ymax"), ("Back (z min)", "zmin"), ("Front (z max)", "zmax"))
XYZ = (("x", "x"), ("y", "y"), ("z", "z"))
PAD = Field("length", choices=(("Automatic", "auto"),), editable=True)      # or a length
SIDE = Field("choice", SIDES)
AXIS = Field("choice", XYZ, editable=True)                                 # also a direction, e.g. 1, 1, 0
LENGTH, POINT = Field("length"), Field("point")


class Selection:
    """A region; evaluate(mesh, on) gives sorted unique indices."""
    TYPE = None
    fields = ()
    field_kinds = {}                                # {field: Field}, one for each of fields

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls.TYPE:
            _TYPES[cls.TYPE] = cls

    def _mask(self, p, mesh, on):
        raise NotImplementedError

    def evaluate(self, mesh, on="nodes", base_dir=None):
        p = _points(mesh, on)
        return np.flatnonzero(self._mask(p, mesh, on)).astype(int)

    def to_dict(self):
        d = {"type": self.TYPE}
        for f in self.fields:
            v = getattr(self, f)
            d[f] = v.tolist() if isinstance(v, np.ndarray) else v
        return d

    def __repr__(self):
        args = ", ".join(f"{f}={getattr(self, f)!r}" for f in self.fields)
        return f"{self.TYPE}({args})"


def from_dict(d):
    """A Selection from its JSON form."""
    d = dict(d)
    kind = d.pop("type", None)
    if kind not in _TYPES:
        raise ValueError(f"unknown selection type {kind!r}; known: {', '.join(sorted(_TYPES))}")
    cls = _TYPES[kind]
    if cls in (Union, Intersect, Difference):
        return cls(*[from_dict(s) for s in d["of"]])
    return cls(**d)


# ----------------------------------------------------------------------------- shapes
class box(Selection):
    """Points with lo <= p <= hi; the corners may come in any order. By default pad is half the element
    size along each axis, so a box thinner than an element still catches the nearest layer."""
    TYPE, fields = "box", ("lo", "hi", "pad")
    field_kinds = {"lo": POINT, "hi": POINT, "pad": PAD}

    def __init__(self, lo, hi, pad="auto"):
        a, b = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        self.lo, self.hi, self.pad = np.minimum(a, b).tolist(), np.maximum(a, b).tolist(), pad

    def _mask(self, p, mesh, on):
        t = _pad(mesh, self.pad, on, axis="each")
        return np.all((p >= np.array(self.lo) - t) & (p <= np.array(self.hi) + t), axis=1)


class cylinder(Selection):
    """Points whose distance to the axis is in [inner_radius, radius]; along = (t0, t1) limits (p - center) . axis."""
    TYPE, fields = "cylinder", ("center", "axis", "radius", "inner_radius", "along", "pad")
    field_kinds = {"center": POINT, "axis": AXIS, "radius": LENGTH, "inner_radius": LENGTH,
                   "along": Field("length", optional=True), "pad": PAD}

    def __init__(self, center, axis, radius, inner_radius=0.0, along=None, pad="auto"):
        self.center = list(map(float, center))
        self.axis = axis if isinstance(axis, str) else list(map(float, axis))
        self.radius, self.inner_radius = _keep(radius), _keep(inner_radius)
        self.along = None if along is None else [_keep(along[0]), _keep(along[1])]
        self.pad = pad

    def _mask(self, p, mesh, on):
        t = _pad(mesh, self.pad, on)
        d = p - np.array(self.center)
        if isinstance(self.axis, str):                 # exact arithmetic for x, y, z axes
            k = _AXES[self.axis]
            i, j = [a for a in range(3) if a != k]
            along = d[:, k]
            r = np.sqrt(d[:, i] ** 2 + d[:, j] ** 2)
        else:
            a = _axis_vector(self.axis)
            along = d @ a
            r = np.linalg.norm(d - np.outer(along, a), axis=1)
        radius, inner = _length(mesh, self.radius), _length(mesh, self.inner_radius)
        m = r <= radius + t
        if inner > 0:
            m &= r >= inner - t
        if self.along is not None:
            a0, a1 = _length(mesh, self.along[0]), _length(mesh, self.along[1])
            m &= (along >= a0 - t) & (along <= a1 + t)
        return m


class sphere(Selection):
    """Points within radius of center."""
    TYPE, fields = "sphere", ("center", "radius", "pad")
    field_kinds = {"center": POINT, "radius": LENGTH, "pad": PAD}

    def __init__(self, center, radius, pad="auto"):
        self.center, self.radius, self.pad = list(map(float, center)), _keep(radius), pad

    def _mask(self, p, mesh, on):
        r = np.sqrt(np.sum((p - np.array(self.center)) ** 2, axis=1))
        return r <= _length(mesh, self.radius) + _pad(mesh, self.pad, on)


class plane(Selection):
    """Points with |p_axis - value| <= pad; by default pad is half the element size along axis.
    limits, e.g. {"y": [100, 120]}, keeps only the part of the cut between those values of the
    other axes (each widened by pad along its own axis)."""
    TYPE, fields = "plane", ("axis", "value", "pad", "limits")
    field_kinds = {"axis": AXIS, "value": LENGTH, "pad": PAD, "limits": Field("limits", optional=True)}

    def __init__(self, axis, value, pad="auto", limits=None):
        self.axis, self.value, self.pad = axis, float(value), pad
        self.limits = None
        if limits:
            bad = [a for a in limits if a not in _AXES or a == axis]
            if bad:
                raise ValueError(f"limits are for the other axes than {axis!r}, not {bad}")
            self.limits = {a: sorted(map(float, ab)) for a, ab in limits.items()}

    def _mask(self, p, mesh, on):
        k = _AXES[self.axis]
        mask = np.abs(p[:, k] - self.value) <= _pad(mesh, self.pad, "nodes", axis=k)
        for a, (lo, hi) in (self.limits or {}).items():
            j = _AXES[a]
            t = _pad(mesh, self.pad, "nodes", axis=j)
            mask &= (p[:, j] >= lo - t) & (p[:, j] <= hi + t)
        return mask

    def to_dict(self):
        d = super().to_dict()
        if d["limits"] is None:                                     # files without limits stay as they were
            del d["limits"]
        return d


def _side(side):
    if len(side) != 4 or side[0] not in _AXES or side[1:] not in ("min", "max"):
        raise ValueError(f"side must be one of xmin, xmax, ymin, ymax, zmin, zmax, not {side!r}")
    return _AXES[side[0]], side[1:]


class face(Selection):
    """The plane at the mesh's extreme on one side: xmin, xmax, ymin, ymax, zmin or zmax."""
    TYPE, fields = "face", ("side", "pad")
    field_kinds = {"side": SIDE, "pad": PAD}

    def __init__(self, side, pad="auto"):
        _side(side)
        self.side, self.pad = side, pad

    def _mask(self, p, mesh, on):
        k, end = _side(self.side)
        extreme = p[:, k].min() if end == "min" else p[:, k].max()
        return np.abs(p[:, k] - extreme) < _pad(mesh, self.pad, "nodes")


class near_face(Selection):
    """Points strictly within distance of the mesh's extreme on one side."""
    TYPE, fields = "near_face", ("side", "distance")
    field_kinds = {"side": SIDE, "distance": LENGTH}

    def __init__(self, side, distance):
        _side(side)
        self.side, self.distance = side, _keep(distance)

    def _mask(self, p, mesh, on):
        k, end = _side(self.side)
        d = _length(mesh, self.distance)
        if end == "min":
            return p[:, k] < p[:, k].min() + d
        return p[:, k] > p[:, k].max() - d


class nearest(Selection):
    """The single node or element closest to point; boundary=True only looks at nodes on
    the mesh boundary, as the PyTO-AD GUI does for a picked point."""
    TYPE, fields = "nearest", ("point", "boundary")
    field_kinds = {"point": POINT, "boundary": Field("tick")}

    def __init__(self, point, boundary=False):
        self.point = list(map(float, point))
        self.boundary = bool(boundary)

    def evaluate(self, mesh, on="nodes", base_dir=None):
        p = _points(mesh, on)
        if not self.boundary:
            return np.array([int(np.argmin(np.sum((p - np.array(self.point)) ** 2, axis=1)))])
        if on != "nodes":
            raise ValueError("nearest with boundary=True selects nodes only")
        from scipy.spatial import cKDTree
        keep = np.asarray(mesh.get_boundary_nodes())
        _, i = cKDTree(p[keep]).query(np.asarray(self.point, dtype=float).reshape(1, 3))
        return np.array([int(keep[int(i[0])])])


class ids(Selection):
    """Explicit node or element numbers; tied to one mesh."""
    TYPE, fields = "ids", ("indices",)
    field_kinds = {"indices": Field("numbers")}

    def __init__(self, indices):
        self.indices = sorted(set(int(i) for i in np.asarray(indices).ravel()))

    def evaluate(self, mesh, on="nodes", base_dir=None):
        n = len(_points(mesh, on))
        bad = [i for i in self.indices if not 0 <= i < n]
        if bad:
            raise IndexError(f"ids selection has indices outside this mesh's {n} {on}: {bad[:5]}")
        return np.array(self.indices, dtype=int)


class stl_faces(Selection):
    """Boundary nodes on triangles of the STL the mesh was made from: within tol times the
    smallest element size of a triangle. tol None is PyTO-AD's own lookup (0.707); the
    PyTO-AD GUI uses 0.8."""
    TYPE, fields = "stl_faces", ("triangles", "tol")
    field_kinds = {"triangles": Field("numbers"), "tol": Field("number", optional=True)}

    def __init__(self, triangles, tol=None):
        self.triangles = [int(t) for t in triangles]
        self.tol = None if tol is None else float(tol)

    def evaluate(self, mesh, on="nodes", base_dir=None):
        if on != "nodes":
            raise ValueError("stl_faces selects nodes only")
        stl = mesh.stlGeom
        bad = [t for t in self.triangles if t >= stl.stl_n_triangles]
        if bad:
            raise ValueError(f"stl_faces: triangles {bad[:5]} are not in the STL, which has "
                             f"{stl.stl_n_triangles}; the selection was made for a different STL")
        if self.tol is None:
            return np.unique(np.asarray(mesh.get_nodes_on_triangles(self.triangles), dtype=int))
        boundary = np.asarray(mesh.get_boundary_nodes())
        points = np.asarray(mesh.node_xyz, dtype=float)[boundary]
        tol = self.tol * min(mesh.elem_size)
        nodes = set()
        for t in self.triangles:
            nodes.update(boundary[stl.find_points_triangle_distances_vectorized(points, t) < tol].tolist())
        return np.array(sorted(nodes), dtype=int)


class halfspaces(Selection):
    """Points on the inner side of every plane: n . (p - o) <= 1e-12 for each normal n and
    origin o. A box dragged in the PyTO-AD GUI is six such planes. boundary=True keeps only
    nodes on the mesh boundary, as the GUI does."""
    TYPE, fields = "halfspaces", ("normals", "origins", "boundary")
    field_kinds = {"normals": Field("numbers"), "origins": Field("numbers"), "boundary": Field("tick")}

    def __init__(self, normals, origins, boundary=True):
        self.normals = [list(map(float, n)) for n in normals]
        self.origins = [list(map(float, o)) for o in origins]
        self.boundary = bool(boundary)

    def evaluate(self, mesh, on="nodes", base_dir=None):
        p = _points(mesh, on)
        n, o = np.asarray(self.normals, dtype=float), np.asarray(self.origins, dtype=float)
        if self.boundary:
            if on != "nodes":
                raise ValueError("halfspaces with boundary=True selects nodes only")
            keep = np.asarray(mesh.get_boundary_nodes())
            inside = np.all(p[keep] @ n.T - np.sum(n * o, axis=1)[None, :] <= 1e-12, axis=1)
            return np.sort(keep[inside]).astype(int)
        return np.flatnonzero(np.all(p @ n.T - np.sum(n * o, axis=1)[None, :] <= 1e-12, axis=1)).astype(int)


class rows(Selection):
    """The n outermost node rows of a face, counted along an axis: e.g. the 8 highest rows
    of the right face. Rows are node layers, so their height is n element sizes."""
    TYPE, fields = "rows", ("face", "axis", "n", "end")
    field_kinds = {"face": SIDE, "axis": AXIS, "n": Field("number"),
                   "end": Field("choice", (("High end", "max"), ("Low end", "min")))}

    def __init__(self, face, axis, n, end="max"):
        _side(face)
        if axis not in _AXES or end not in ("min", "max"):
            raise ValueError(f"rows: axis must be x, y or z and end 'min' or 'max'")
        self.face, self.axis, self.n, self.end = face, axis, int(n), end

    def evaluate(self, mesh, on="nodes", base_dir=None):
        if on != "nodes":
            raise ValueError("rows selects nodes only")
        p = _points(mesh, "nodes")
        k, side_end = _side(self.face)
        a = _AXES[self.axis]
        h = np.asarray(mesh.elem_size, dtype=float)
        extreme = p[:, k].min() if side_end == "min" else p[:, k].max()
        on_face = np.abs(p[:, k] - extreme) < 0.25 * h[k]
        row = np.round(p[:, a] / h[a]).astype(int)
        chosen = np.unique(row[on_face])
        chosen = chosen[-self.n:] if self.end == "max" else chosen[:self.n]
        return np.flatnonzero(on_face & np.isin(row, chosen)).astype(int)


def _resolve(path, base_dir):
    return path if os.path.isabs(path) or base_dir is None else os.path.join(base_dir, path)


class inside_stl(Selection):
    """Points inside a closed STL surface, the surface first moved by shift (x, y, z in model units), e.g.
    when the STL was saved shifted to 0, 0, 0. shift is written only when it is not zero."""
    TYPE, fields = "inside_stl", ("path", "shift")
    field_kinds = {"path": Field("file", files="STL files (*.stl *.STL)"),
                   "shift": Field("point", help="Moves the shape before nodes are selected, for example when "
                                                "the STL was saved shifted to 0, 0, 0.")}

    def __init__(self, path, shift=(0.0, 0.0, 0.0)):
        self.path = path
        self.shift = [float(v) for v in shift]
        if len(self.shift) != 3:
            raise ValueError(f"shift needs x, y and z, not {shift!r}")

    def surface(self, base_dir=None):
        """The closed surface, moved by shift."""
        import pyvista as pv
        path = _resolve(self.path, base_dir)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"inside_stl: {path} does not exist")
        surface = pv.read(path)
        return surface.translate(self.shift, inplace=False) if any(self.shift) else surface

    def evaluate(self, mesh, on="nodes", base_dir=None):
        import pyvista as pv
        pts = pv.PolyData(_points(mesh, on))
        sel = pts.select_enclosed_points(self.surface(base_dir), tolerance=0.0, check_surface=False)
        return np.flatnonzero(np.asarray(sel.point_data["SelectedPoints"]).astype(bool)).astype(int)

    def to_dict(self):
        d = super().to_dict()
        if not any(d["shift"]):                                     # files without a move stay as they were
            del d["shift"]
        return d

    def __repr__(self):                                             # the code line: shift only when not zero
        args = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items() if k != "type")
        return f"{self.TYPE}({args})"


class function(Selection):
    """Whatever a Python function name(mesh, on) in file.py returns."""
    TYPE, fields = "function", ("ref",)
    field_kinds = {"ref": Field("file", files="Python files (*.py)")}

    def __init__(self, ref):
        if ":" not in ref:
            raise ValueError(f"function selection {ref!r}: write it as 'file.py:name'")
        self.ref = ref

    def evaluate(self, mesh, on="nodes", base_dir=None):
        file, name = self.ref.rsplit(":", 1)
        path = _resolve(file, base_dir)
        spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        out = np.asarray(getattr(module, name)(mesh, on)).ravel()
        if out.dtype == bool:
            out = np.flatnonzero(out)
        return np.unique(out.astype(int))


# ----------------------------------------------------------------------------- combinations
class Union(Selection):
    """Points in any of the parts."""
    TYPE = "union"

    def __init__(self, *parts):
        self.parts = parts

    def evaluate(self, mesh, on="nodes", base_dir=None):
        return np.unique(np.concatenate([s.evaluate(mesh, on, base_dir) for s in self.parts])).astype(int)

    def to_dict(self):
        return {"type": self.TYPE, "of": [s.to_dict() for s in self.parts]}

    def __repr__(self):
        return f"{self.TYPE}({', '.join(map(repr, self.parts))})"


class Intersect(Union):
    """Points in all of the parts."""
    TYPE = "intersect"

    def evaluate(self, mesh, on="nodes", base_dir=None):
        out = self.parts[0].evaluate(mesh, on, base_dir)
        for s in self.parts[1:]:
            out = np.intersect1d(out, s.evaluate(mesh, on, base_dir))
        return out.astype(int)


class Difference(Union):
    """Points in the first part but not the second."""
    TYPE = "difference"

    def __init__(self, a, b):
        super().__init__(a, b)

    def evaluate(self, mesh, on="nodes", base_dir=None):
        a, b = self.parts
        return np.setdiff1d(a.evaluate(mesh, on, base_dir), b.evaluate(mesh, on, base_dir)).astype(int)


class selection(Selection):
    """Another selection of the project, by name. Only a project can evaluate it: Project.evaluate puts
    the named selection in (resolved) each time, so a combination follows later edits of its parts."""
    TYPE, fields = "selection", ("name",)
    field_kinds = {"name": Field("choice")}

    def __init__(self, name):
        self.name = name

    def evaluate(self, mesh, on="nodes", base_dir=None):
        raise ValueError(f"selection({self.name!r}) is looked up by the project: evaluate it with Project.evaluate")


LINKS = ("selection",)                  # the JSON types that name another selection


def _walk(d, f):
    """The JSON form d with f applied to every link to another selection in it."""
    if d.get("type") in LINKS:
        return f(d)
    if "of" in d:
        return dict(d, of=[_walk(s, f) for s in d["of"]])
    return d


def names_in(selection):
    """The names of the other selections a selection refers to."""
    out = []
    _walk(selection.to_dict(), lambda d: out.append(d["name"]) or d)
    return out


def resolved(selection, lookup, seen=()):
    """The selection with every selection(name) replaced by lookup[name], itself resolved; ValueError for a
    name lookup does not have, or a selection that refers back to itself."""
    def put_in(d):
        name = d["name"]
        if name in seen:
            raise ValueError(f"{' -> '.join(seen + (name,))}: a selection cannot contain itself")
        if name not in lookup:
            raise ValueError(f"there is no selection named {name!r}")
        return resolved(lookup[name], lookup, seen + (name,)).to_dict()
    return from_dict(_walk(selection.to_dict(), put_in))


def renamed(selection, old, new):
    """The selection with selection(old) changed to selection(new); written as selection, whatever it said."""
    return from_dict(_walk(selection.to_dict(),
                           lambda d: dict(d, type="selection", name=new) if d["name"] == old else d))


def holds_typed_lengths(selection):
    """True when the selection holds a coordinate or a length typed in model units: a field given as a
    point or a length (field_kinds) whose value is not in element sizes h ("1.5h" is, "5 + 1h" is not;
    so are the words of a choice, such as "auto"). Such values stay as typed when the part is scaled.
    Inside combinations too."""
    def typed(v):
        if isinstance(v, (list, tuple)):
            return any(typed(x) for x in v)
        if isinstance(v, str):
            try:
                return _length_value(v, 0.0) != 0
            except ValueError:
                return False
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if any(holds_typed_lengths(part) for part in getattr(selection, "parts", ())):
        return True
    return any(field.kind in ("point", "length") and typed(getattr(selection, name))
               for name, field in selection.field_kinds.items())


def union(*parts):
    return Union(*parts)


def intersect(*parts):
    return Intersect(*parts)


def difference(a, b):
    return Difference(a, b)
