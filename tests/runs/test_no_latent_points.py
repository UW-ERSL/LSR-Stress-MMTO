"""
test_no_latent_points.py
========================
A run whose material source has no latent points (latent dimension 0): the design is the densities only, and
every step of the run takes it, from the start design and its bounds through the iterations, the final check
and the save. The source here is the database's own values with one material for every element
(materials/material_source.py, DatabaseMaterials), put in where the run makes its source
(optimization_loop.material_source).

    python -m pytest tests/runs/test_no_latent_points.py
"""
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

MATERIAL = 2                                          # the row every element is given


def _one_material(monkeypatch, made):
    """The run's material source replaced by the database's own values, MATERIAL for every element, with no
    latent points; made["source"] is the source the run got, made["vector"] the size of each design it asked
    the model about."""
    from lsr_mmto.materials.material_source import DatabaseMaterials
    from lsr_mmto.optimize import design_model, optimization_loop as M

    def source(s, vae_params, xlsx, saveNet, should_stop, on_network_error):
        made["source"] = DatabaseMaterials(xlsx, MATERIAL)
        return made["source"], torch.zeros((len(made["source"].materialNames), 0)), {}, saveNet
    monkeypatch.setattr(M, "material_source", source)
    real = design_model.MMTOModelAD.evaluate

    def evaluate(self, zeta, *args, **kwargs):
        made.setdefault("vector", set()).add(np.asarray(zeta).size)
        return real(self, zeta, *args, **kwargs)
    monkeypatch.setattr(design_model.MMTOModelAD, "evaluate", evaluate)


def test_a_run_with_no_latent_points_goes_through(monkeypatch, capsys):
    """The U-bracket on a coarse mesh, its distance penalty and latent smoothing switched on as usual: the run asks
    about designs of the densities only (one number per element), leaves out both penalties (they read the latent
    points) and says so, goes through and saves; each element has MATERIAL's database values, the saved run has
    latent dimension 0 and no history of either penalty, and every element's material is MATERIAL."""
    from lsr_mmto.optimize import save_results
    from lsr_mmto.problem.project import Project, run
    from lsr_mmto.results import load_results
    made, saved = {}, {}
    _one_material(monkeypatch, made)
    real_save = save_results.save_run_data

    def keep(**kw):
        saved.update(kw["fields"])
        return real_save(**kw)
    monkeypatch.setattr(save_results, "save_run_data", keep)
    p = Project.load(os.path.join(ROOT, "projects", "U-bracket.mmto"))
    p.n_dof = 3000
    p.settings["common"]["maxIterations"] = 3
    p.settings["advanced"]["use_penalization"] = True
    p.settings["advanced"]["use_z_smoothing"] = True
    with tempfile.TemporaryDirectory() as d:
        r = load_results.load(os.path.join(run(p, save_dir=d, verbose=False), "run_data"))
    assert "No latent points: left out Latent smoothing, Distance penalty" in capsys.readouterr().out
    assert not {"Pz_n", "Pn", "P"} & set(r.history), sorted(r.history)
    source = made["source"]
    assert made["vector"] == {r.n_elems}, made["vector"]
    assert r.n_latent == 0 and r.z_elem.shape == (r.n_elems, 0)
    assert np.all(r.material_indices == MATERIAL)
    E = source.rawData[MATERIAL, source.materialAttributes["Youngs_Modulus"]["idx"]]
    assert np.allclose(saved["property_Youngs_Modulus"], E)
    assert np.all(np.isfinite(r.history["J_phys"])) and r.iterations >= 1
