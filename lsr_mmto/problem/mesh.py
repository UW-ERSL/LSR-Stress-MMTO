"""
mesh.py
=======
The voxel mesh of a part: make_mesh(stl_path, nDOFDesired, single_layer, unknowns_per_node), the
hexahedral elements inside the STL, as many as make about nDOFDesired unknowns for a physics with
that many unknowns at each node (nDOFDesired / unknowns_per_node elements asked for; 3 for structural,
the displacement in x, y and z). Every project's build() meshes its part with it. Each physics' solver
makes the element-to-unknown table it needs (PyTO-AD's structural and thermal solvers both do).
"""
import numpy as np

from lsr_mmto.pyto_ad_imports import hex_mesher


def make_mesh(stl_path, nDOFDesired, single_layer=False, unknowns_per_node=3):
    """The voxel mesh of an STL, with nDOFDesired / unknowns_per_node elements asked for."""
    mesh = hex_mesher.HexMesher()
    if single_layer:
        mesh.minVoxelsPerAxis = 1
    mesh.createMeshFromSTLFile(stl_path, nElemsDesired=nDOFDesired / unknowns_per_node)
    mesh.elem_size = np.asarray(mesh.elem_size, dtype=float)   # an array, not a list: Numba-compiled code
    return mesh                                                # stops taking lists in a coming version
