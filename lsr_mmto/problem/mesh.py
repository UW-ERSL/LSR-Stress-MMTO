"""
mesh.py
=======
The voxel mesh of a part: make_mesh(stl_path, nDOFDesired, single_layer), the hexahedral
elements inside the STL, with nDOFDesired / 3 elements asked for. Every project's build()
meshes its part with it.
"""
import numpy as np

from lsr_mmto.pyto_ad_imports import hex_mesher


def make_mesh(stl_path, nDOFDesired, single_layer=False):
    """The voxel mesh of an STL, with nDOFDesired / 3 elements asked for."""
    mesh = hex_mesher.HexMesher()
    if single_layer:
        mesh.minVoxelsPerAxis = 1
    mesh.createMeshFromSTLFile(stl_path, nElemsDesired=nDOFDesired / 3)
    mesh.createEdofMatStructural()
    mesh.elem_size = np.asarray(mesh.elem_size, dtype=float)   # an array, not a list: Numba-compiled code
    return mesh                                                # stops taking lists in a coming version
