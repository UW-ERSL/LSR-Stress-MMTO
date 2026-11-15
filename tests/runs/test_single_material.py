"""
test_single_material.py
=======================
A project of a single material, every element the same material, the design the densities only: one database
material (material_source "one_database_material": pinned_material, or the database's only material, at its
database values, no network read or trained), or the network pinned to one database material
(pinned_material, that material's properties as the network gives them). Each runs through to the save; the project saves and reads back its
choice, an older file reads as the network with no pin, and the project's check refuses what cannot run.

    python -m pytest tests/runs/test_single_material.py
"""
import json
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

DATABASE = os.path.join(ROOT, "data", "databases", "6Materials_AF_3Attr")
ROW = 2                                               # the material each test makes single


def _one_material_database(folder, row=ROW):
    """A database of one material, the 6-material database's row, written in folder; returns its path."""
    from openpyxl import Workbook
    from lsr_mmto.materials.databases import database_table
    names, attributes, units, values = database_table(DATABASE + ".xlsx")
    wb = Workbook()
    ws = wb.active
    ws.append(["Attributes"] + list(attributes))
    ws.append(["Material/Units"] + list(units))
    ws.append([str(names[row])] + [float(v) for v in values[row]])
    path = os.path.join(folder, "one_material.xlsx")
    wb.save(path)
    return path


def _ubracket(iterations=3):
    """The shipped U-bracket on a coarse mesh, a few iterations."""
    from lsr_mmto.problem.project import Project
    p = Project.load(os.path.join(ROOT, "projects", "U-bracket.mmto"))
    p.n_dof = 3000
    p.settings["common"]["maxIterations"] = iterations
    return p


def _saved_run(p, folder, **options):
    """p run into folder (with run's options); (the run read back, the arrays the run saved)."""
    from lsr_mmto.optimize import save_results
    from lsr_mmto.problem.project import run
    from lsr_mmto.results import load_results
    saved, real = {}, save_results.save_run_data

    def keep(**kw):
        saved.update(kw["fields"])
        return real(**kw)
    save_results.save_run_data = keep
    try:
        return load_results.load(os.path.join(run(p, save_dir=folder, verbose=False, **options), "run_data")), saved
    finally:
        save_results.save_run_data = real


def test_one_database_material_runs_with_no_network():
    """A database of one material and material_source "one_database_material": the project passes its check, the
    run reads and trains no network (no .nt is written beside the database), leaves out the penalties of the
    latent points, and every element has the material's database values; the saved run has latent dimension 0."""
    from lsr_mmto.materials.databases import database_table
    with tempfile.TemporaryDirectory() as d:
        p = _ubracket()
        p.database = _one_material_database(d)
        p.material_source = "one_database_material"
        assert p.validate() == [], p.validate()
        r, saved = _saved_run(p, os.path.join(d, "run"))
        assert not os.path.exists(os.path.join(d, "one_material.nt"))
    _, attributes, _, values = database_table(DATABASE + ".xlsx")
    E = values[ROW, attributes.index("Youngs_Modulus")]
    assert r.n_latent == 0 and np.all(r.material_indices == 0)
    assert np.allclose(saved["property_Youngs_Modulus"], E)
    assert not {"Pz_n", "Pn", "P"} & set(r.history)


def test_one_material_of_a_database_of_several():
    """Single material with the 6-material database itself: the material named (pinned_material) is every
    element's, at its database values, with no network read or trained; nothing needs deleting from the
    database."""
    from lsr_mmto.materials.databases import database_table
    names, attributes, _, values = database_table(DATABASE + ".xlsx")
    p = _ubracket()
    p.material_source, p.pinned_material = "one_database_material", str(names[ROW])
    assert p.validate() == [], p.validate()
    with tempfile.TemporaryDirectory() as d:
        r, saved = _saved_run(p, os.path.join(d, "run"))
    assert r.n_latent == 0 and np.all(r.material_indices == ROW) and r.meta["vae_net"] is None
    assert np.allclose(saved["property_Youngs_Modulus"], values[ROW, attributes.index("Youngs_Modulus")])


def test_the_settings_of_the_latent_points_change_nothing_with_one_material():
    """Every setting that says it reads the latent points (Setting.reads_latent_points, which the Settings panel
    greys out with one material) changed from its default: a run of one database material, and a run of the
    network fixed to one material, are the same, to the last digit."""
    from lsr_mmto.problem.settings import SETTINGS
    changed = {"use_penalization": False, "gamma_init": 1.0, "gamma_max": 100.0, "gamma_factor": 2.0,
               "penalty_start_iter": 1, "distance_penalty_disable_after_iter": 2,
               "use_dynamic_gamma_scheduling": False, "gamma_plateau_window": 2, "gamma_plateau_tolerance": 0.5,
               "gamma_plateau_earliest_iter": 1, "gamma_ramp_end_margin": 1, "use_z_smoothing": False,
               "z_smoothing_weight": 1.0, "z_smoothing_disable_after_iter": 1, "material_filter_radius": 5.0,
               "z0_init_method": "UNIFORM", "snap_to_real_material": True}
    assert set(changed) == {s.key for s in SETTINGS if s.reads_latent_points}
    from lsr_mmto.materials.databases import database_table
    for single in (True, False):
        runs = []
        with tempfile.TemporaryDirectory() as d:
            for edit in ({}, changed):
                p = _ubracket()
                if single:
                    p.database = _one_material_database(d)
                    p.material_source = "one_database_material"
                else:
                    p.pinned_material = str(database_table(DATABASE + ".xlsx")[0][ROW])
                p.complete_settings()
                for key, value in edit.items():
                    section = next(s.section for s in SETTINGS if s.key == key)
                    p.settings[section][key] = value
                runs.append(_saved_run(p, os.path.join(d, f"run{len(runs)}"))[0])
        a, b = runs
        assert len(a.history["objective"]) == 4 and np.array_equal(a.history["objective"], b.history["objective"])
        assert np.array_equal(a.x_cont, b.x_cont) and np.array_equal(a.material_indices, b.material_indices)


def test_a_material_name_with_spaces_around_it():
    """A database whose material's name has spaces around it (a cell typed "C "): the network reads the name as
    the project's check does, without them, so the network fixed to "C" runs and gives that row; and the
    record of a run of one material says it left out the penalties and, with no network, has no network
    settings."""
    import shutil
    from openpyxl import load_workbook
    from lsr_mmto.materials.databases import database_table
    name = str(database_table(DATABASE + ".xlsx")[0][ROW])
    with tempfile.TemporaryDirectory() as d:
        book = load_workbook(DATABASE + ".xlsx")
        book.worksheets[0].cell(row=3 + ROW, column=1).value = f" {name} "
        book.save(os.path.join(d, "spaced.xlsx"))
        shutil.copy(DATABASE + ".nt", os.path.join(d, "spaced.nt"))
        p = _ubracket()
        p.database = os.path.join(d, "spaced.xlsx")
        p.pinned_material = name
        assert p.validate() == [], p.validate()
        r, _ = _saved_run(p, os.path.join(d, "run"))
        assert r.n_latent == 0 and np.all(r.material_indices == ROW) and r.material_names[ROW] == name
        p = _ubracket()
        p.database = _one_material_database(d)
        p.material_source = "one_database_material"
        r, _ = _saved_run(p, os.path.join(d, "single"))
    assert r.config["vae_params"] is None and r.meta["vae_net"] is None
    assert r.config["penalization_distance"]["use_penalization"] is False
    assert r.config["penalization_zsmoothing"]["use_z_smoothing"] is False


def test_the_network_pinned_to_one_material():
    """pinned_material names a material of the 6-material database: every element has that material's properties
    as the shipped network gives them at the material's own latent point, the design is the densities only, and
    every element's material is that one, in the saved run and in what each iteration tells the window."""
    from lsr_mmto.materials.databases import DATABASE_VAE, VAEParams, database_table
    from lsr_mmto.materials.encoder import MaterialEncoder
    names = [str(n) for n in database_table(DATABASE + ".xlsx")[0]]
    p = _ubracket()
    p.pinned_material = names[ROW]
    assert p.validate() == [], p.validate()
    heard = []
    with tempfile.TemporaryDirectory() as d:
        r, saved = _saved_run(p, d, progress=lambda state: heard.append(state["material"]))
    assert heard and all(np.all(m == ROW) and len(m) == r.n_elems for m in heard)
    params = VAEParams()
    for k, v in DATABASE_VAE["6Materials_AF_3Attr"].items():
        setattr(params, k, v)
    enc = MaterialEncoder(params)
    enc.readExcel(DATABASE + ".xlsx")
    enc.loadAutoencoderFromFile(DATABASE + ".nt")
    with torch.no_grad():
        E = float(enc.decode(torch.as_tensor(enc.training_latents[ROW:ROW + 1], dtype=torch.float64))
                  ["Youngs_Modulus"][0])
    assert r.n_latent == 0 and np.all(r.material_indices == ROW)
    assert np.allclose(saved["property_Youngs_Modulus"], E)


def test_the_choice_is_saved_and_an_older_file_reads_as_the_network():
    """material_source and pinned_material are saved and read back; a file saved before them reads as the network
    with no pin; the project script sets them only when they are not a new project's."""
    from lsr_mmto.problem.project import Project
    with tempfile.TemporaryDirectory() as d:
        p = _ubracket()
        p.pinned_material = "anything"
        path = os.path.join(d, "u.mmto")
        p.save(path)
        q = Project.load(path)
        assert (q.material_source, q.pinned_material) == ("network", "anything")
        script = open(q.save_as_script(os.path.join(d, "u_setup.py"))).read()
        assert "p.pinned_material = 'anything'" in script and "p.material_source" not in script
        data = json.load(open(path))
        del data["mmto"]["material_source"], data["mmto"]["pinned_material"]
        json.dump(data, open(path, "w"))
        old = Project.load(path)
        assert (old.material_source, old.pinned_material) == ("network", None)


def test_what_cannot_run_is_refused():
    """The project's check refuses: a material source it does not know; a pin that names no material of the
    database, with either source; single material with no material chosen from a database of several (one
    material, none chosen, is that one); the network with a database of one material; and, for a single
    material, a quantity that reads the latent points (material volume fraction)."""
    from lsr_mmto.materials.databases import database_table

    def said(p):
        return " | ".join(p.validate())
    p = _ubracket()
    p.material_source = "something"
    assert "Materials in the design: 'something' is not a choice LSR MMTO knows" in said(p)
    p = _ubracket()
    p.pinned_material = "no such material"
    assert "Fix every element to: 'no such material' is not in" in said(p)
    p = _ubracket()
    p.material_source = "one_database_material"
    assert "Single material: choose the material, one of the 6 in 6Materials_AF_3Attr.xlsx" in said(p)
    p.pinned_material = "anything"
    assert "Material: 'anything' is not in 6Materials_AF_3Attr.xlsx" in said(p)
    with tempfile.TemporaryDirectory() as d:
        p = _ubracket()
        p.database = _one_material_database(d)
        assert "Multi material: one_material.xlsx has one material, and the network needs at least two" in said(p)
        p.material_source = "one_database_material"
        assert p.validate() == [], p.validate()
    p = _ubracket()
    p.pinned_material = str(database_table(DATABASE + ".xlsx")[0][ROW])
    p.constraint("MaterialVolumeFraction(1)", "<=", 0.3)
    assert "needs a mix of materials, and this design has one material" in said(p)
