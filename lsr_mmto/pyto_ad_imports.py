"""
pyto_ad_imports.py
==================
Every name LSR MMTO takes from PyTO-AD, in one place. PyTO-AD is the folder PyTO-AD beside
this repository; importing this module puts its src folder on sys.path, and warns about what
is known to cause trouble (PyTO-AD missing, numpy too new, torch missing). It never stops
anything.
"""
import os
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def check_environment(root: str) -> None:
    """Warn about what is known to cause trouble. Packages are not imported."""
    warnings = []

    pyto_ad = os.path.join(os.path.dirname(root), "PyTO-AD", "src")
    if not os.path.isdir(pyto_ad):
        warnings.append(f"PyTO-AD not found at {pyto_ad}. Run setup_environment.py, "
                        "or see Setting up by hand in the README.")

    try:
        numpy_version = version("numpy")
        major, minor = (int(v) for v in re.findall(r"\d+", numpy_version)[:2])
        if (major, minor) >= (2, 3):
            warnings.append(f"numpy {numpy_version} is installed; this code is tested "
                            "with numpy below 2.3 (see requirements.txt).")
    except PackageNotFoundError:
        warnings.append("numpy is not installed in this environment.")

    try:
        version("torch")
    except PackageNotFoundError:
        warnings.append("torch is not installed in this environment; this branch "
                        "needs it for the gradients.")

    for w in warnings:
        print(f"[lsr_mmto] warning: {w}")


check_environment(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../PyTO-AD/src')))
import pyto.autodiff.material_model as topopt_material_model  # first: PyTO-AD sets torch's default dtype to float64
from pyto.core import hex_mesher, mat_lib
from pyto.core import bc as bound_cond
from pyto.topopt import filters as topopt_filters
from pyto.physics.structural import hex_structural_fea
from pyto.physics import hex_element_stiffness
from pyto.physics.thermal import hex_thermal_fea
from pyto.solve import solvers as linear_solvers
from pyto.solve import deflation

from pyto.topopt.common import createFilters, TOParams, find_elements_with_fixedDOF, DIRECT_SOLVER_DOF_CUTOFF
from pyto.topopt.drivers.mmaWrapper import runMMA
from pyto.autodiff.material_model import MaterialModel, get_pNorm_exponent
from pyto.autodiff.material_model import initialize_SIMP_STRUCTURAL_PENALTY, initialize_SIMP_THERMAL_PENALTY
from pyto.autodiff.material_model import increment_SIMP_THERMAL_PENALTY, increment_SIMP_STRUCTURAL_PENALTY

import pyto
PYTO_AD_DIR = os.path.dirname(os.path.abspath(pyto.__file__))   # the PyTO-AD package in use, for the run record


def stl_recovery() -> "ModuleType":
    """PyTO-AD's STL recovery (pyto.io.topopt_stl_recovery), imported when first asked for: it takes
    a third of a second and only Export STL needs it."""
    from pyto.io import topopt_stl_recovery
    return topopt_stl_recovery
