"""
mesh.py
=======
The voxel mesh of a part: make_mesh(stl_path, nDOFDesired, single_layer, unknowns_per_node), the
hexahedral elements inside the STL, as many as make about nDOFDesired unknowns for a physics with
that many unknowns at each node (nDOFDesired / unknowns_per_node elements asked for; 3 for structural,
the displacement in x, y and z). Every project's build() meshes its part with it. Each physics' solver
makes the element-to-unknown table it needs (PyTO-AD's structural and thermal solvers both do).
surface_faces(mesh): the faces of the mesh's surface, with their outward normals and areas, for loads
that act on the surface (a pressure along the normal, a heat flux, convection).
"""
from typing import TYPE_CHECKING

import numpy as np

from lsr_mmto.pyto_ad_imports import hex_mesher

if TYPE_CHECKING:
    from pyto.core.hex_mesher import HexMesher


def make_mesh(stl_path: str, nDOFDesired: int, single_layer: bool = False,
              unknowns_per_node: int = 3) -> "HexMesher":
    """The voxel mesh of an STL, with nDOFDesired / unknowns_per_node elements asked for."""
    mesh = hex_mesher.HexMesher()
    if single_layer:
        mesh.minVoxelsPerAxis = 1
    mesh.createMeshFromSTLFile(stl_path, nElemsDesired=nDOFDesired / unknowns_per_node)
    mesh.elem_size = np.asarray(mesh.elem_size, dtype=float)   # an array, not a list: Numba-compiled code
    return mesh                                                # stops taking lists in a coming version


# the six faces of a hexahedral element as its four corner nodes, each in the order PyTO-AD's mesher uses
# (hex_mesher.get_boundary_surface_quads): -z, +z, -y, +y, -x, +x
HEX_FACES = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [3, 7, 6, 2], [0, 4, 7, 3], [1, 2, 6, 5]])


def surface_faces(mesh: "HexMesher") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(quads, normals, areas) of the voxel mesh's surface: every element face that belongs to one
    element only, as its four nodes (an array of shape (F, 4)), its outward unit normal (F, 3) and its
    area (F,). Voxel faces are axis-aligned, so each normal is one of +-x, +-y, +-z. All faces at once,
    with numpy (PyTO-AD's get_boundary_surface_quads loops over the elements in Python, and returns no
    face for these meshes)."""
    elems = np.asarray(mesh.elemArray, dtype=np.int64)
    quads = elems[:, HEX_FACES].reshape(-1, 4)                              # (6 elements, 4), element by element
    owner = np.repeat(np.arange(len(elems)), 6)
    _, first, count = np.unique(np.sort(quads, axis=1), axis=0, return_index=True, return_counts=True)
    keep = np.sort(first[count == 1])                                       # a face of one element only
    quads, owner = quads[keep], owner[keep]
    xyz = np.asarray(mesh.node_xyz, dtype=float)
    out = xyz[quads].mean(axis=1) - xyz[elems[owner]].mean(axis=1)          # face centre minus element centre
    axis = np.argmax(np.abs(out), axis=1)
    rows = np.arange(len(out))
    normals = np.zeros_like(out)
    normals[rows, axis] = np.sign(out[rows, axis])
    h = np.asarray(mesh.elem_size, dtype=float)
    areas = np.prod(h) / h[axis]
    return quads, normals, areas
