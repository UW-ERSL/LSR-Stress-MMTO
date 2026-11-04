"""
test_selections.py
==================
Each selection type on a 10 x 10 x 10 grid of unit cubes, where the answer is
known by hand: nodes at integer coordinates 0..10, element centers at 0.5..9.5.
No solver is needed, so this runs in seconds.

    python tests/test_selections.py        (or: python -m pytest tests)
"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lsr_mmto.problem import selections as S  # noqa: E402


class GridMesh:
    """The attributes a selection reads from PyTO-AD's HexMesher."""
    def __init__(self, n=10):
        g = np.arange(n + 1, dtype=float)
        self.node_xyz = np.array([[x, y, z] for z in g for y in g for x in g])
        c = np.arange(n, dtype=float) + 0.5
        self.elem_centers = np.array([[x, y, z] for z in c for y in c for x in c])
        self.elem_size = [1.0, 1.0, 1.0]
        self.n = n

    def get_boundary_nodes(self):
        on = np.any((self.node_xyz == 0) | (self.node_xyz == self.n), axis=1)
        return np.flatnonzero(on)


MESH = GridMesh()


def count(sel, on="nodes", **kw):
    return len(sel.evaluate(MESH, on, **kw))


def test_every_field_says_how_it_is_given():
    """Each selection type declares a Field for each of its fields: the GUI's form is made from them."""
    for name, cls in S._TYPES.items():
        assert set(cls.field_kinds) == set(cls.fields), name
        for field, how in cls.field_kinds.items():
            assert isinstance(how, S.Field), (name, field)
            assert how.kind in ("length", "point", "choice", "tick", "file", "number", "numbers", "limits"), \
                (name, field, how.kind)


def test_box():
    assert count(S.box([0, 0, 0], [2, 2, 2])) == 27              # coordinates 0, 1, 2 in each direction
    assert count(S.box([0, 0, 0], [2, 2, 2]), "elements") == 8   # centers 0.5, 1.5; no pad on centers
    assert count(S.box([0, 0, 0], [2, 2, 2], pad="half"), "elements") == 27   # forced: 2.5 comes in
    assert count(S.box([0.2, 0, 0], [2, 2, 2], pad=0)) == 18     # x = 0 now outside


def test_face_plane_near_face():
    assert count(S.face("xmin")) == 121
    assert count(S.face("zmax")) == 121
    assert count(S.face("xmin"), "elements") == 100               # the outermost layer only
    assert np.array_equal(S.face("zmin").evaluate(MESH), S.plane("z", 0.0).evaluate(MESH))
    plate = GridMesh()                                            # 1.418 wide in x, 1 thick, as the L-bracket
    plate.node_xyz = plate.node_xyz * [1.418, 1.0, 1.0]
    plate.elem_size = [1.418, 1.418, 1.0]
    for x in np.linspace(0.0, 14.18, 41):                        # every x in the part: the nearest column
        assert len(S.plane("x", x).evaluate(plate)) >= 121, x
    assert count(S.plane("x", 3, limits={"y": [2, 4]})) == 3 * 11   # y 2, 3, 4; every z
    assert S.plane("x", 3).to_dict() == {"type": "plane", "axis": "x", "value": 3.0, "pad": "auto"}
    assert count(S.box([2, 2, 2], [0, 0, 0])) == 27               # corners in any order
    for x in np.linspace(0.0, 14.18, 41):                        # a box thinner than an element: a column
        assert len(S.box([x, 0, 0], [x, 10, 10]).evaluate(plate)) >= 121, x
    assert count(S.near_face("xmin", 2.0), "elements") == 200    # centers 0.5 and 1.5
    assert count(S.near_face("xmax", 2.0), "elements") == 200


def test_selections_by_name():
    u = S.union(S.selection("a"), S.face("xmax"))
    assert S.names_in(u) == ["a"]
    assert np.array_equal(S.resolved(u, {"a": S.face("xmin")}).evaluate(MESH),
                          S.union(S.face("xmin"), S.face("xmax")).evaluate(MESH))
    assert S.names_in(S.renamed(u, "a", "b")) == ["b"]
    for lookup, says in (({"a": S.union(S.selection("a"))}, "cannot contain itself"), ({}, "no selection named")):
        try:
            S.resolved(u, lookup)
        except ValueError as e:
            assert says in str(e), e
        else:
            raise AssertionError(says)


def test_cylinder():
    c = dict(center=[5, 5, 0], axis="z", pad=0)
    assert count(S.cylinder(radius=1.0, **c)) == 5 * 11          # (5,5) and its 4 neighbours, 11 layers
    assert count(S.cylinder(radius=1.0, along=(0, 2), **c)) == 5 * 3
    assert count(S.cylinder(radius=1.5, inner_radius=1.0, **c)) == 8 * 11   # r = 1 and r = sqrt 2
    # a vector axis gives the same as the named one
    v = S.cylinder(center=[5, 5, 0], axis=[0, 0, 2], radius=1.5, inner_radius=1.0, pad=0)
    assert np.array_equal(v.evaluate(MESH), S.cylinder(radius=1.5, inner_radius=1.0, **c).evaluate(MESH))


def test_sphere_nearest_ids():
    assert count(S.sphere([5, 5, 5], 1.0, pad=0)) == 7
    i = S.nearest([0.2, 0.1, 0.4]).evaluate(MESH)
    assert len(i) == 1 and np.allclose(MESH.node_xyz[i[0]], [0, 0, 0])
    assert list(S.ids([5, 3, 3]).evaluate(MESH)) == [3, 5]
    try:
        S.ids([10 ** 6]).evaluate(MESH)
    except IndexError:
        pass
    else:
        raise AssertionError("ids outside the mesh did not raise")


def test_element_size_lengths_and_rows():
    """'2h' is two element sizes; rows() takes the outermost node rows of a face."""
    assert count(S.near_face("xmin", "2h"), "elements") == 200
    assert count(S.sphere([5, 5, 5], "1h", pad=0)) == 7
    assert count(S.cylinder([5, 5, 0], "z", "1h", along=(0, "2h"), pad=0)) == 5 * 3
    assert count(S.rows("xmax", "y", 3)) == 3 * 11                # y = 8, 9, 10 on x = 10
    assert count(S.rows("xmax", "y", 3, end="min")) == 3 * 11
    top = S.rows("ymax", "x", 2).evaluate(MESH)
    assert set(MESH.node_xyz[top, 0]) == {9.0, 10.0} and set(MESH.node_xyz[top, 1]) == {10.0}
    for bad in (lambda: S.near_face("xmin", "2mm"), lambda: S.rows("xmax", "w", 2)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("a bad length or axis was accepted")


def test_lengths_in_element_sizes():
    """A length may be arithmetic in numbers, pi and h, the largest element size: kept as text when it has
    h and worked out on each mesh; without h it is a number at once. "1.5h" means what it always did."""
    class Coarse(GridMesh):                                         # the same grid, elements twice as big
        def __init__(self):
            super().__init__()
            self.elem_size = [2.0, 2.0, 2.0]
    coarse = Coarse()
    for text, fine, big in (("5 + 1h", 6.0, 7.0), ("5 + h", 6.0, 7.0), ("2*h", 2.0, 4.0), ("2 h", 2.0, 4.0),
                            ("38/2 + 0.5h", 19.5, 20.0), ("1.5h", 1.5, 3.0), ("h", 1.0, 2.0), ("pi*h", np.pi, 2 * np.pi)):
        assert S._keep(text) == text, text                          # kept as typed
        assert S._length(MESH, text) == fine and S._length(coarse, text) == big, text
    for h in (0.7, 1.3, 0.12345678):                                # "1.5h" bit for bit as before
        mesh = type("M", (), {"elem_size": [h, h / 2, h]})()
        assert S._length(mesh, "1.5h") == float("1.5") * h and S._length(mesh, "h") == 1.0 * h
    assert S._keep("2 + 2") == 4.0 and isinstance(S._keep("38/2"), float)   # no h: a number at once
    s = S.sphere([5, 5, 5], "1 + 1h", pad=0)
    assert s.to_dict()["radius"] == "1 + 1h" and S.from_dict(json.loads(json.dumps(s.to_dict()))).radius == "1 + 1h"
    assert count(s) == len(S.sphere([5, 5, 5], 2.0, pad=0).evaluate(MESH))
    assert len(s.evaluate(coarse)) == len(S.sphere([5, 5, 5], 3.0, pad=0).evaluate(MESH)) > count(s)
    assert S.near_face("xmin", "1.5h").to_dict() == {"type": "near_face", "side": "xmin", "distance": "1.5h"}
    for bad in ("5 + x", "2mm", "abs(h)", "h.real", "e*h", "1/0", "h +", "hh"):
        try:
            S._keep(bad)
        except ValueError as e:
            assert "element sizes h" in str(e), (bad, e)
        else:
            raise AssertionError(f"accepted {bad!r}")


def test_stl_faces():
    """stl_faces asks the mesh for the nodes on those triangles (PyTO-AD's own lookup)."""
    class FakeMesh(GridMesh):
        stlGeom = type("STL", (), {"stl_n_triangles": 100})()

        def get_nodes_on_triangles(self, tris):
            return [5, 3, 3] if list(tris) == [12, 13] else []
    assert list(S.stl_faces([12, 13]).evaluate(FakeMesh())) == [3, 5]
    assert S.from_dict(S.stl_faces([12, 13]).to_dict()).triangles == [12, 13]
    try:
        S.stl_faces([200]).evaluate(FakeMesh())
    except ValueError as e:
        assert "not in the STL" in str(e)
    else:
        raise AssertionError("a triangle outside the STL was accepted")


def test_gui_box_and_point():
    """A box dragged in the GUI (six planes) and a picked point, on boundary nodes only."""
    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([2.0, 2.0, 2.0])
    normals = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
    origins = [lo, hi, lo, hi, lo, hi]
    box = S.halfspaces(normals, origins)
    inside = set(S.box(lo, hi, pad=0).evaluate(MESH).tolist())
    boundary = set(MESH.get_boundary_nodes().tolist())
    assert set(box.evaluate(MESH).tolist()) == inside & boundary
    assert count(box) == 27 - 8              # the 8 nodes with every coordinate 1 or 2 are inside the part
    assert count(S.halfspaces(normals, origins, boundary=False)) == 27
    p = S.nearest([5.2, 5.1, 4.6], boundary=True).evaluate(MESH)        # an interior point
    assert MESH.node_xyz[p[0]].tolist() in ([5, 5, 0], [5, 5, 10], [5, 0, 5], [0, 5, 5], [10, 5, 5], [5, 10, 5])
    assert S.nearest([5.2, 5.1, 4.6]).evaluate(MESH).tolist() != p.tolist()


def test_combinations():
    a, b = S.face("xmin"), S.face("ymin")
    assert count(S.union(a, b)) == 121 + 121 - 11
    assert count(S.intersect(a, b)) == 11                         # the edge x = y = 0
    assert count(S.difference(a, b)) == 121 - 11
    # Akshat's load: top face, within 2 of x = 5
    top_middle = S.intersect(S.face("ymax"), S.box([3, 0, 0], [7, 10, 10], pad=0))
    assert count(top_middle) == 5 * 11


def test_json_round_trip():
    sels = [S.box([0, 0, 0], [2, 2, 2]), S.cylinder([5, 5, 0], "z", 1.5, 1.0, (0, 2), pad=0),
            S.cylinder([5, 5, 0], [0, 0, 1], 1.0), S.sphere([5, 5, 5], 1.0), S.plane("y", 3.0),
            S.face("xmax"), S.near_face("zmin", 2.0), S.near_face("zmax", "1.5h"), S.nearest([1, 2, 3]),
            S.ids([1, 2]), S.rows("xmax", "y", 3), S.nearest([1, 2, 3], boundary=True),
            S.halfspaces([[1, 0, 0]], [[2, 0, 0]]), S.halfspaces([[1, 0, 0]], [[2, 0, 0]], boundary=False),
            S.union(S.face("xmin"), S.intersect(S.face("ymin"), S.difference(S.face("zmin"), S.ids([0]))))]
    for s in sels:
        d = json.loads(json.dumps(s.to_dict()))
        t = S.from_dict(d)
        assert t.to_dict() == s.to_dict(), (s, t)
        node_only = s.TYPE == "rows" or getattr(s, "boundary", False)
        for on in ("nodes", "elements") if not node_only else ("nodes",):
            assert np.array_equal(t.evaluate(MESH, on), s.evaluate(MESH, on)), (s, on)
    try:
        S.from_dict({"type": "cone"})
    except ValueError as e:
        assert "unknown selection type" in str(e)
    else:
        raise AssertionError("unknown type did not raise")


def test_function_and_inside_stl():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "bcs.py"), "w") as fh:
            fh.write("def corner(mesh, on):\n    return mesh.node_xyz.sum(axis=1) == 0\n")
        s = S.function("bcs.py:corner")
        assert list(s.evaluate(MESH, base_dir=d)) == [0]
        assert S.from_dict(s.to_dict()).ref == "bcs.py:corner"

        import pyvista as pv
        pv.Cube(center=(3, 3, 3), x_length=2, y_length=2, z_length=2).triangulate().save(os.path.join(d, "c.stl"))
        inside = S.inside_stl("c.stl")
        assert count(inside, "elements", base_dir=d) == 8         # centers 2.5 and 3.5
        assert np.array_equal(inside.evaluate(MESH, "elements", base_dir=d),
                              S.box([2, 2, 2], [4, 4, 4], pad=0).evaluate(MESH, "elements"))
        assert inside.to_dict() == {"type": "inside_stl", "path": "c.stl"} and repr(inside) == "inside_stl(path='c.stl')"

        # the same cube saved shifted to 0, 0, 0, moved back by Move by: the same nodes and elements
        pv.Cube(center=(0, 0, 0), x_length=2, y_length=2, z_length=2).triangulate().save(os.path.join(d, "c0.stl"))
        moved = S.inside_stl("c0.stl", shift=[3, 3, 3])
        for on in ("nodes", "elements"):
            assert len(inside.evaluate(MESH, on, base_dir=d))
            assert np.array_equal(moved.evaluate(MESH, on, base_dir=d), inside.evaluate(MESH, on, base_dir=d))
        assert not np.array_equal(S.inside_stl("c0.stl").evaluate(MESH, "nodes", base_dir=d),
                                  inside.evaluate(MESH, "nodes", base_dir=d))
        assert moved.to_dict() == {"type": "inside_stl", "path": "c0.stl", "shift": [3.0, 3.0, 3.0]}
        assert S.from_dict(moved.to_dict()).to_dict() == moved.to_dict()
        assert repr(moved) == "inside_stl(path='c0.stl', shift=[3.0, 3.0, 3.0])"
        assert S.from_dict({"type": "inside_stl", "path": "c.stl"}).shift == [0.0, 0.0, 0.0]   # old files


if __name__ == "__main__":
    failed = 0
    for name, fn in [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]:
        try:
            fn()
            print(f"ok   {name}")
        except Exception as e:                                     # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)
