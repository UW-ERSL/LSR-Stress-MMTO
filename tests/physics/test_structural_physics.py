"""
test_structural_physics.py
==========================
The structural physics (lsr_mmto/physics/structural/): its tiny example, which tests/runs/test_every_physics_runs.py
checks and runs. Structural is the physics the core's own tests use, so its other tests are with the core's: a run
and its record in tests/runs/, its gradients in tests/gradients/, the paper's examples in
tests/runs/test_regression.py. The smallest test file a physics can have: start from this one for a new physics.

    python -m pytest tests/physics/test_structural_physics.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

BAR = os.path.join(ROOT, "data", "models", "RectangularBeam")


def tiny_example(folder):
    """The smallest structural project that runs: the bar on a coarse mesh, held on its xmin face and pulled down
    on its xmax face, its compliance made least with half its volume, the 6-material database and its network,
    2 iterations. folder is a scratch folder (this example writes nothing there)."""
    from lsr_mmto.problem import selections as S
    from lsr_mmto.problem.project import Project
    p = Project("tiny structural", stl="RectangularBeam.STL", n_dof=1500, single_layer=True, base_dir=BAR)
    p.physics = "structural"
    p.database = "6Materials_AF_3Attr"
    p.add_selection("xmin", S.face("xmin"))
    p.add_selection("xmax", S.face("xmax"))
    p.add_bc("fixed", "xmin", dofs="xyz")
    p.add_bc("force", "xmax", total=[0, -1, 0])
    p.objective("Compliance()")
    p.constraint("VolumeFraction()", "<=", 0.5)
    p.settings["common"]["maxIterations"] = 2
    return p

