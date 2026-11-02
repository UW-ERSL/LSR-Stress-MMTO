"""
test_final_check.py
===================
The final check at the end of a run (compliance, stresses, failure factor of the continuous,
binarized and snapped designs) must analyse the design with the same materials as the
optimization: each element's Young's modulus from the network, and its Poisson's ratio either from
the network or, when the project says so, the constant one.

It once used 0.3 (steel's) for every element whatever the project said: the beam, with a constant
0.37, got a max failure factor 7% too high.

Two short coarse runs: the beam (constant Poisson's ratio 0.37) and the L-bracket (Poisson's ratio
from the network). About a minute.

    python tests/test_final_check.py        (or: python -m pytest tests)
"""
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lsr_mmto.optimize import final_check as U  # noqa: E402
from lsr_mmto.problem.project import Project, run  # noqa: E402

N_DOF, ITERATIONS = 3000, 2


def final_checks(name):
    """Run the project briefly; for each final check: the E and nu the solver used, the E and nu
    the project asks for (worked out here from the network directly), and the check's compliance."""
    p = Project.load(os.path.join(ROOT, "projects", f"{name}.mmto"))
    p.n_dof = N_DOF
    p.settings["common"]["maxIterations"] = ITERATIONS
    advanced = p.settings["advanced"]
    constant = bool(advanced["use_constant_poissons_ratio"])
    nu_constant = float(advanced["constant_poissons_ratio"])
    seen, real = [], U.eval_stress_compliance_metrics

    def check(**kw):
        result = real(**kw)
        fe, enc = kw["fe_solver_structural"], kw["matEncoder"]
        lam, mu = fe.lam.detach().numpy(), fe.mu.detach().numpy()
        nu_used = lam / (2.0 * (lam + mu))
        E_used = 2.0 * mu * (1.0 + nu_used)
        z = torch.tensor(np.asarray(kw["z_vec"], float).reshape(kw["latentDim"], -1).T,
                         dtype=torch.get_default_dtype())
        with torch.no_grad():
            props = enc.getMaterialProperties(enc.vaeNet.decoder(z))
        E = props["Youngs_Modulus"].numpy()
        nu = np.full_like(E, nu_constant) if constant else props["Poissons_Ratio"].numpy()
        seen.append({"label": kw["label"], "E_used": E_used, "nu_used": nu_used, "E": E, "nu": nu,
                     "compliance": result["compliance"]})
        return result

    U.eval_stress_compliance_metrics = check
    try:
        with tempfile.TemporaryDirectory() as d:
            run(p, save_dir=d, verbose=False)
    finally:
        U.eval_stress_compliance_metrics = real
    return seen


def _same_materials(name):
    checks = final_checks(name)
    assert len(checks) == 3, f"{name}: {len(checks)} final checks, expected continuous, binarized, snapped"
    for c in checks:
        assert np.allclose(c["E_used"], c["E"], rtol=1e-9), f"{name}, {c['label']}: Young's modulus"
        assert np.allclose(c["nu_used"], c["nu"], rtol=1e-9, atol=1e-12), \
            f"{name}, {c['label']}: Poisson's ratio {c['nu_used'].min():.4g}..{c['nu_used'].max():.4g}, " \
            f"the project's {c['nu'].min():.4g}..{c['nu'].max():.4g}"
        assert np.isfinite(c["compliance"]) and c["compliance"] > 0, f"{name}, {c['label']}: compliance"
    return checks


def test_constant_poissons_ratio_is_used():
    checks = _same_materials("beam")
    assert np.allclose(checks[0]["nu"], 0.37), "the beam project is meant to use a constant 0.37"


def test_poissons_ratio_from_the_network_is_used():
    checks = _same_materials("L-bracket")
    assert not np.allclose(checks[0]["nu"], 0.3), "the L-bracket's materials should not all be 0.3"


if __name__ == "__main__":
    test_constant_poissons_ratio_is_used()
    test_poissons_ratio_from_the_network_is_used()
    print("ok")
