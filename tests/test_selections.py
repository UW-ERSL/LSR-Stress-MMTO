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
sys.path.insert(0, os.path.join(ROOT, "0-Common"))
import selections as S  # noqa: E402


class GridMesh:
    """The attributes a selection reads from PyTO-AD's HexMesher."""
    def __init__(self, n=10):
        g = np.arange(n + 1, dtype=float)
        self.node_xyz = np.array([[x, y, z] for z in g for y in g for x in g])
        c = np.arange(n, dtype=float) + 0.5
        self.elem_centers = np.array([[x, y, z] for z in c for y in c for x in c])
        self.elem_size = [1.0, 1.0, 1.0]


MESH = GridMesh()


def count(sel, on="nodes", **kw):
    return len(sel.evaluate(MESH, on, **kw))


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
    assert count(S.near_face("xmin", 2.0), "elements") == 200    # centers 0.5 and 1.5
    assert count(S.near_face("xmax", 2.0), "elements") == 200


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
            S.face("xmax"), S.near_face("zmin", 2.0), S.nearest([1, 2, 3]), S.ids([1, 2]),
            S.union(S.face("xmin"), S.intersect(S.face("ymin"), S.difference(S.face("zmin"), S.ids([0]))))]
    for s in sels:
        d = json.loads(json.dumps(s.to_dict()))
        t = S.from_dict(d)
        assert t.to_dict() == s.to_dict(), (s, t)
        for on in ("nodes", "elements"):
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
