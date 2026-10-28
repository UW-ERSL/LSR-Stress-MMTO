import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../PyTO-AD/src')))
import pyto.autodiff.material_model as topopt_material_model  # first: PyTO-AD sets torch's default dtype to float64
from pyto.io import stl_reader
from pyto.core import hex_mesher, mat_lib
from pyto.core import bc as bound_cond
from pyto.topopt import filters as topopt_filters
from pyto.physics.structural import hex_structural_fea
from pyto.physics import hex_element_stiffness
from pyto.physics.thermal import hex_thermal_fea
from pyto.solve import solvers as linear_solvers
from pyto.solve import deflation

from pyto.topopt.common import createFilters, TO_QOI, TOParams, find_elements_with_fixedDOF, DIRECT_SOLVER_DOF_CUTOFF
from pyto.topopt.drivers.mmaWrapper import runMMA
from pyto.autodiff.material_model import MaterialModel, get_pNorm_exponent
from pyto.autodiff.material_model import initialize_SIMP_STRUCTURAL_PENALTY, initialize_SIMP_THERMAL_PENALTY
from pyto.autodiff.material_model import increment_SIMP_THERMAL_PENALTY, increment_SIMP_STRUCTURAL_PENALTY
