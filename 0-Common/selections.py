"""
selections.py
=============
Named regions of a voxel mesh: which nodes carry a support or a load, which elements
are kept solid. A selection is a description (a box, a cylinder, a face, ...), not a
list of indices, so it holds for any mesh size and can be saved in a project file.

    from selections import box, cylinder, face, union
    left = face("xmin")
    bore = cylinder(center=[0.1, 0, 9.5], axis="y", radius=6.0, along=(-3.5, 3.0), pad=0)
    nodes = left.evaluate(mesh, on="nodes")          # sorted node indices
    elems = bore.evaluate(mesh, on="elements")       # sorted element indices (by centers)

on="nodes" tests the mesh nodes, on="elements" the element centers.

pad widens every bound, in model units. The default "auto" is half the smallest element
size for nodes, so a face at x = 0 still catches nodes at x = 1e-12, and 0 for element
centers, which sit half an element from any face and would otherwise pull in the next
layer. plane and face always take half an element by default, since they need a
width: on elements, face gives the outermost layer. "half" forces half an element;
pad=0 is exact.

Types (to_dict / from_dict give the JSON form a project file stores):
    box(lo, hi)                                  lo <= p <= hi
    cylinder(center, axis, radius, inner_radius=0, along=None)
                                                 distance to the axis in [inner_radius, radius];
                                                 along = (t0, t1) limits (p - center) . axis
    sphere(center, radius)
    plane(axis, value)                           |p_axis - value| < pad
    face(side)                                   the plane at the mesh's extreme, side "xmin" ... "zmax"
    near_face(side, distance)                    within distance of that extreme (strictly)
    nearest(point)                               the single closest node or element
    ids(indices)                                 explicit indices; tied to one mesh
    inside_stl(path)                             inside a closed STL surface
    function("file.py:name")                     name(mesh, on) returns the indices
    union(a, b, ...), intersect(a, b, ...), difference(a, b)

A function selection runs the Python file it names when it is evaluated: open project
files with function selections only from people you trust.

To add a type: subclass Selection, give it a TYPE and a fields tuple, write _mask or
evaluate, and add it to tests/test_selections.py. from_dict finds it by TYPE.
"""
import importlib.util
import os

import numpy as np

_AXES = {"x": 0, "y": 1, "z": 2}
_TYPES = {}


def _points(mesh, on):
    if on == "nodes":
        return np.asarray(mesh.node_xyz, dtype=float)
    if on == "elements":
        return np.asarray(mesh.elem_centers, dtype=float)
    raise ValueError(f"on must be 'nodes' or 'elements', not {on!r}")


def _pad(mesh, pad, on):
    if pad == "auto":
        pad = "half" if on == "nodes" else 0.0
    if pad == "half":
        return 0.5 * float(np.min(mesh.elem_size))
    return float(pad)


def _axis_vector(axis):
    if isinstance(axis, str):
        v = np.zeros(3)
        v[_AXES[axis]] = 1.0
        return v
    v = np.asarray(axis, dtype=float)
    return v / np.linalg.norm(v)


class Selection:
    """A region; evaluate(mesh, on) gives sorted unique indices."""
    TYPE = None
    fields = ()

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
    TYPE, fields = "box", ("lo", "hi", "pad")

    def __init__(self, lo, hi, pad="auto"):
        self.lo, self.hi, self.pad = list(map(float, lo)), list(map(float, hi)), pad

    def _mask(self, p, mesh, on):
        t = _pad(mesh, self.pad, on)
        return np.all((p >= np.array(self.lo) - t) & (p <= np.array(self.hi) + t), axis=1)


class cylinder(Selection):
    TYPE, fields = "cylinder", ("center", "axis", "radius", "inner_radius", "along", "pad")

    def __init__(self, center, axis, radius, inner_radius=0.0, along=None, pad="auto"):
        self.center = list(map(float, center))
        self.axis = axis if isinstance(axis, str) else list(map(float, axis))
        self.radius, self.inner_radius = float(radius), float(inner_radius)
        self.along = None if along is None else [float(along[0]), float(along[1])]
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
        m = r <= self.radius + t
        if self.inner_radius > 0:
            m &= r >= self.inner_radius - t
        if self.along is not None:
            m &= (along >= self.along[0] - t) & (along <= self.along[1] + t)
        return m


class sphere(Selection):
    TYPE, fields = "sphere", ("center", "radius", "pad")

    def __init__(self, center, radius, pad="auto"):
        self.center, self.radius, self.pad = list(map(float, center)), float(radius), pad

    def _mask(self, p, mesh, on):
        r = np.sqrt(np.sum((p - np.array(self.center)) ** 2, axis=1))
        return r <= self.radius + _pad(mesh, self.pad, on)


class plane(Selection):
    TYPE, fields = "plane", ("axis", "value", "pad")

    def __init__(self, axis, value, pad="auto"):
        self.axis, self.value, self.pad = axis, float(value), pad

    def _mask(self, p, mesh, on):
        return np.abs(p[:, _AXES[self.axis]] - self.value) < _pad(mesh, self.pad, "nodes")


def _side(side):
    if len(side) != 4 or side[0] not in _AXES or side[1:] not in ("min", "max"):
        raise ValueError(f"side must be one of xmin, xmax, ymin, ymax, zmin, zmax, not {side!r}")
    return _AXES[side[0]], side[1:]


class face(Selection):
    TYPE, fields = "face", ("side", "pad")

    def __init__(self, side, pad="auto"):
        _side(side)
        self.side, self.pad = side, pad

    def _mask(self, p, mesh, on):
        k, end = _side(self.side)
        extreme = p[:, k].min() if end == "min" else p[:, k].max()
        return np.abs(p[:, k] - extreme) < _pad(mesh, self.pad, "nodes")


class near_face(Selection):
    TYPE, fields = "near_face", ("side", "distance")

    def __init__(self, side, distance):
        _side(side)
        self.side, self.distance = side, float(distance)

    def _mask(self, p, mesh, on):
        k, end = _side(self.side)
        if end == "min":
            return p[:, k] < p[:, k].min() + self.distance
        return p[:, k] > p[:, k].max() - self.distance


class nearest(Selection):
    TYPE, fields = "nearest", ("point",)

    def __init__(self, point):
        self.point = list(map(float, point))

    def evaluate(self, mesh, on="nodes", base_dir=None):
        p = _points(mesh, on)
        return np.array([int(np.argmin(np.sum((p - np.array(self.point)) ** 2, axis=1)))])


class ids(Selection):
    TYPE, fields = "ids", ("indices",)

    def __init__(self, indices):
        self.indices = sorted(set(int(i) for i in np.asarray(indices).ravel()))

    def evaluate(self, mesh, on="nodes", base_dir=None):
        n = len(_points(mesh, on))
        bad = [i for i in self.indices if not 0 <= i < n]
        if bad:
            raise IndexError(f"ids selection has indices outside this mesh's {n} {on}: {bad[:5]}")
        return np.array(self.indices, dtype=int)


def _resolve(path, base_dir):
    return path if os.path.isabs(path) or base_dir is None else os.path.join(base_dir, path)


class inside_stl(Selection):
    TYPE, fields = "inside_stl", ("path",)

    def __init__(self, path):
        self.path = path

    def evaluate(self, mesh, on="nodes", base_dir=None):
        import pyvista as pv
        path = _resolve(self.path, base_dir)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"inside_stl: {path} does not exist")
        pts = pv.PolyData(_points(mesh, on))
        sel = pts.select_enclosed_points(pv.read(path), tolerance=0.0, check_surface=False)
        return np.flatnonzero(np.asarray(sel.point_data["SelectedPoints"]).astype(bool)).astype(int)


class function(Selection):
    TYPE, fields = "function", ("ref",)

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
    TYPE = "intersect"

    def evaluate(self, mesh, on="nodes", base_dir=None):
        out = self.parts[0].evaluate(mesh, on, base_dir)
        for s in self.parts[1:]:
            out = np.intersect1d(out, s.evaluate(mesh, on, base_dir))
        return out.astype(int)


class Difference(Union):
    TYPE = "difference"

    def __init__(self, a, b):
        super().__init__(a, b)

    def evaluate(self, mesh, on="nodes", base_dir=None):
        a, b = self.parts
        return np.setdiff1d(a.evaluate(mesh, on, base_dir), b.evaluate(mesh, on, base_dir)).astype(int)


def union(*parts):
    return Union(*parts)


def intersect(*parts):
    return Intersect(*parts)


def difference(a, b):
    return Difference(a, b)
