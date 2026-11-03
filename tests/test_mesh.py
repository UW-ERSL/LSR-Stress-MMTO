"""
test_mesh.py
============
The voxel mesh's surface: surface_faces gives every element face that belongs to one element only, with
its outward normal and its area. On a box the faces cover the six sides exactly, their areas add up to the
box's surface, the area-weighted normals add up to zero, as on any closed surface, and each side has the
faces of its grid.

    python tests/test_mesh.py        (or: python -m pytest tests)
"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lsr_mmto.problem.mesh import make_mesh, surface_faces  # noqa: E402

SIZE = (40.0, 10.0, 5.0)


def box_mesh():
    import pyvista as pv
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "box.stl")
        pv.Box(bounds=(0, SIZE[0], 0, SIZE[1], 0, SIZE[2])).triangulate().save(path)
        return make_mesh(path, 3000)


def test_surface_faces_of_a_box():
    mesh = box_mesh()
    quads, normals, areas = surface_faces(mesh)
    lo, hi = mesh.node_xyz.min(0), mesh.node_xyz.max(0)
    side = hi - lo
    assert np.isclose(areas.sum(), 2 * (side[0] * side[1] + side[1] * side[2] + side[0] * side[2]))
    assert np.allclose((areas[:, None] * normals).sum(0), 0.0, atol=1e-9 * areas.sum())
    centre = mesh.node_xyz.mean(0)
    face_centres = mesh.node_xyz[quads].mean(1)
    assert np.all(np.einsum("ij,ij->i", face_centres - centre, normals) > 0)      # every normal points out
    on_side = np.isclose(face_centres, lo) | np.isclose(face_centres, hi)
    assert np.all(on_side.any(axis=1))                                            # every face lies on a side


def test_a_box_has_the_faces_of_its_grid():
    """A box meshed as an nx by ny by nz grid of elements has 2 (nx ny + ny nz + nz nx) surface faces, each
    side's faces with that side's normal."""
    mesh = box_mesh()
    quads, normals, _ = surface_faces(mesh)
    n = np.rint((mesh.node_xyz.max(0) - mesh.node_xyz.min(0)) / np.asarray(mesh.elem_size)).astype(int)
    assert int(np.prod(n)) == mesh.num_elems                                       # the box is a full grid
    assert len(quads) == 2 * (n[0] * n[1] + n[1] * n[2] + n[2] * n[0])
    for axis in range(3):
        across = [n[i] for i in range(3) if i != axis]
        for sign in (-1, 1):
            assert np.sum(normals[:, axis] == sign) == across[0] * across[1]


if __name__ == "__main__":
    test_surface_faces_of_a_box()
    test_a_box_has_the_faces_of_its_grid()
    print("ok")
