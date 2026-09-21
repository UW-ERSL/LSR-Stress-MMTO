import enum
import sys
import os
import numpy as np
# Add the 0-Common folder to Python path
common_path = os.path.join(os.path.dirname(__file__), '..', '0-Common')
sys.path.insert(0, os.path.abspath(common_path))
from materialColors import material_colors # type: ignore[reportMissingImports]
# Now you can import normally
from StructuralFEAExamples import StructuralFEAExamples, getStructuralFEAProblem # type: ignore
from PyTOImports import  TO_QOI, find_elements_with_fixedDOF # type: ignore
from mmto_common import MMTOParams # type: ignore
from dataclasses import dataclass, field
import topopt_material_model  # to override the shared p-norm exponent per problem
@dataclass(slots=True) # avoid accidental modification
class VAEParams: #default VAE parameters
    klFactor: float = 5e-6
    learningRate: float = 2e-6
    numEpochs: int = 200000 # max epochs
    vae_hiddenDim: int = 500
    latentDim: int = 2 # default latent dimension
    maxAttributeErrorPercent: float = 1 # termination criteria for VAE training

class MMTOExamplesPureStructural(enum.Enum):
    LBracketTopLoad_Stress_BM1 = enum.auto()
    Beam_Liao_Pareto_FF = enum.auto()
    AirplaneBracket_Stress = enum.auto()
    UBracket_Stress_FF = enum.auto()

# --- Beam_Liao_Pareto_FF sweep overrides: set these from the sweep loop before
#     each run_topopt call (mirrors the topopt_material_model._PNORM_EXPONENT pattern).
#     Mass cap is in solver units (rho[kg/m^3] * volume[mm^3], mesh-invariant);
#     traction sets the load (total quarter force = traction * 88.515 mm^2). ---
_BEAM_PARETO_MASS_CAP = 4.9e8   # default = mid Pareto point
_BEAM_PARETO_TRACTION = 150.0   # placeholder; calibrate so loosest-mass run gives max FF ~ 0.4

# --- U-bracket FF benchmark knobs (unit square minus top slot; nz=1). Tune after a test run,
#     mirroring the _BEAM_PARETO_* pattern. Mass = integral(rho dV) is mesh-invariant. ---
_UBRACKET_LOAD          = 1.0     # total -x force over the top-right node rows (unit load)
_UBRACKET_DOMAIN_AREA   = 0.875   # unit square (1.0) minus slot (0.25*0.5 = 0.125)
_UBRACKET_THICKNESS     = 0.01    # z-extent baked into UBracket.STL (single nz=1 layer)
_UBRACKET_MASS_FRACTION = 0.40    # cap = 40% of an A-filled (rho=1.0) domain; raise/lower to bind
_UBRACKET_MASS_CAP      = _UBRACKET_MASS_FRACTION * _UBRACKET_DOMAIN_AREA * _UBRACKET_THICKNESS  # rho_A=1.0


def getMMTOProblemPureStructural(to_problem: MMTOExamplesPureStructural,nDOFDesired = None, **kwargs):
    """Get the structural topology optimization problem based on the specified example.

    Args:
        problem: The example problem to solve.
        **kwargs: Additional arguments to pass to the problem.

    Returns:
        StructuralTOProblem: The structural topology optimization problem.
    """
    
    to_params = MMTOParams()
    topopt_material_model._PNORM_EXPONENT = 6   # default; Xie branches override to 10
    vae_params = VAEParams()

    if to_problem == MMTOExamplesPureStructural.LBracketTopLoad_Stress_BM1:

        structural_problem = StructuralFEAExamples.LBracket_BM1
        kwargs['topload'] = 0
        kwargs['midload'] = 4

        to_params.ExtrudeZ = True
        to_params.RelativeFilterRadius = 3.5
        # to_params.MaterialFilterRadius = 1.5 # smaller filter radius for material distribution can help capture fine features in the material distribution, which is important for criticality-based constraints
        to_params.nDOFDesired = 30000 if nDOFDesired is None else nDOFDesired

        to_params.MaterialsExcelFile = './1-PureStructural/MaterialDataPureStructural/BM1_3Mats_2Attr.xlsx'

        to_params.Objective = (TO_QOI.PNORM_STRESS, None)
        # to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.40)]
        to_params.Constraints = [
            (TO_QOI.VOLUME_FRACTION, {"material_id":0, "tau":0.5}, 0.15),
            (TO_QOI.VOLUME_FRACTION, {"material_id":1, "tau":0.5}, 0.15),
            (TO_QOI.VOLUME_FRACTION, {"material_id":2, "tau":0.5}, 0.10),
        ]

        vae_params.latentDim = 2
        vae_params.learningRate = 5e-4
        vae_params.vae_hiddenDim = 50
        vae_params.maxAttributeErrorPercent = 0.1
        vae_params.klFactor = 1e-4
        vae_params.numEpochs = 30000

    elif to_problem == MMTOExamplesPureStructural.Beam_Liao_Pareto_FF:
        structural_problem = StructuralFEAExamples.Beam_Liao
        to_params.Comment = ("Liao supported beam (quarter), 8-material VAE, min p-norm "
                             "FAILURE FACTOR s.t. mass cap; mass swept for Pareto (no cost)")
        to_params.Objective = (TO_QOI.PNORM_STRESS, "FAILURE_FACTOR")
        to_params.RelativeFilterRadius = 7     # Liao R_phi=30mm at ~5.3mm elem; mesh-consistent
        to_params.nDOFDesired = 80000 if nDOFDesired is None else nDOFDesired
        topopt_material_model._PNORM_EXPONENT = 10
        to_params.MaterialsExcelFile = './1-PureStructural/MaterialDataPureStructural/Beam_8Mats_3Attr_v4_spectrum2D.xlsx'
        # single mass cap, no cost / no volume-fraction caps; swept via _BEAM_PARETO_MASS_CAP
        to_params.Constraints = [
            (TO_QOI.MASS, None, _BEAM_PARETO_MASS_CAP),
        ]
        # load set by _BEAM_PARETO_TRACTION (threaded to createBeam_Liao_Problem via kwargs)
        kwargs['traction'] = _BEAM_PARETO_TRACTION
        vae_params.latentDim = 2
        vae_params.learningRate = 5e-4
        vae_params.vae_hiddenDim = 50
        vae_params.maxAttributeErrorPercent = 0.1
        vae_params.klFactor = 1e-4
        vae_params.numEpochs = 100000
    elif to_problem == MMTOExamplesPureStructural.AirplaneBracket_Stress:
            structural_problem = StructuralFEAExamples.AirplaneBracket
            to_params.Comment = "Airplane bracket (Liao 2024 geometry), z-half, 20-material VAE, min p-norm FAILURE FACTOR s.t. mass cap, GE-style non-design keep"
            to_params.ExtrudeZ = False
            to_params.RelativeFilterRadius = 7.3       # tune; raise if thin members appear
            to_params.KeepFixedElems = False           # keep clamped bolt-hole elements solid
            to_params.nDOFDesired = 150000 if nDOFDesired is None else nDOFDesired   # ~h=1.35mm; bump to ~124000 for ~150k DOF
            topopt_material_model._PNORM_EXPONENT = 10
            # 20-material DB in mm-N-MPa-tonne units (E/yield in MPa, density in tonne/mm^3)
            to_params.MaterialsExcelFile = './1-PureStructural/MaterialDataPureStructural/CustomAlloys20_3D.xlsx'
            # objective: minimize p-norm of per-element failure factor sigma_vm,e / Y_e
            to_params.Objective = (TO_QOI.PNORM_STRESS, "FAILURE_FACTOR")
            # single mass cap (tonne). Sweep {1.0e-4, 1.5e-4, 2.0e-4}; all-Ti-10V-solid ~= 2.9e-4 is the topology threshold
            to_params.Constraints = [
                (TO_QOI.MASS, None, 4e7),   # ~35% of a Ti-density-filled domain (kg/m^3 * mm^3 units)
                (TO_QOI.COST, None, 7.5e8),   # USD/kg * kg/m^3 * mm^3 ; tighten -> more material diversity
            ]
            vae_params.latentDim = 3
            vae_params.learningRate = 5e-4
            vae_params.vae_hiddenDim = 50
            vae_params.maxAttributeErrorPercent = 0.1
            vae_params.klFactor = 1e-4
            vae_params.numEpochs = 100000

    elif to_problem == MMTOExamplesPureStructural.UBracket_Stress_FF:
        structural_problem = StructuralFEAExamples.UBracket
        kwargs['load'] = _UBRACKET_LOAD
        to_params.Comment = ("U-bracket (unit square minus top slot), nz=1, 6-material A-F VAE, "
                             "min p-norm FAILURE FACTOR s.t. mass cap (-x distributed load)")
        to_params.ExtrudeZ = True            # nz=1 single layer (extrude filter is identity)
        to_params.RelativeFilterRadius = 2.5
        to_params.nDOFDesired = 30000 if nDOFDesired is None else nDOFDesired
        topopt_material_model._PNORM_EXPONENT = 16
        to_params.MaterialsExcelFile = './1-PureStructural/MaterialDataPureStructural/6Materials_AF_3Attr.xlsx'
        # objective: minimize p-norm of per-element failure factor sigma_vm,e / Y_e
        to_params.Objective = (TO_QOI.PNORM_STRESS, "FAILURE_FACTOR")
        # single mass cap, no cost / no volume-fraction caps; tune via _UBRACKET_MASS_FRACTION
        to_params.Constraints = [(TO_QOI.MASS, None, _UBRACKET_MASS_CAP)]
        vae_params.latentDim = 2
        vae_params.learningRate = 1e-4
        vae_params.vae_hiddenDim = 50
        vae_params.maxAttributeErrorPercent = 0.1
        vae_params.klFactor = 1e-4
        vae_params.numEpochs = 100000
    else:
        raise ValueError(f"Unknown problem: {to_problem}")
    
    mesh, mat_prop, bc, elem_body_force = getStructuralFEAProblem(structural_problem, nDOFDesired = to_params.nDOFDesired, **kwargs)


    # Add  elements to keep
    to_params.ElemsToKeep  = None # default value

    # Here we add additional parameters specific to the optimization problem
    if (to_params.KeepFixedElems):
        to_params.ElemsToKeep = find_elements_with_fixedDOF(mesh, bc,nDOFPerNode=3)
    # GE-style non-design: union the NonDesign collar into ElemsToKeep.
    # This only biases the DENSITY block toward solid (MainPureStructural.py:608);
    # the latent/material block is left free. Fires only for problems that set
    # mesh.nonDesignElems, so other examples are unaffected.
    if getattr(mesh, "nonDesignElems", None) is not None and len(mesh.nonDesignElems) > 0:
        if to_params.ElemsToKeep is None:
            to_params.ElemsToKeep = mesh.nonDesignElems
        else:
            to_params.ElemsToKeep = np.union1d(to_params.ElemsToKeep, mesh.nonDesignElems)

    return mesh,mat_prop, bc, elem_body_force, to_params, vae_params

if __name__ == "__main__":
    # Temperature independent Mult-Material TO Problems :
    
    to_problem = MMTOExamplesPureStructural.CantileverBenchmark_Compliance_Mass
    mesh, mat_prop, bc, elem_body_force, to_params, vae_params = getMMTOProblemPureStructural(to_problem, nDOFDesired=30000)
    print(f"Obtained problem: {to_problem.name}.")