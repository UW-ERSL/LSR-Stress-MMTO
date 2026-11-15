"""
test_network_stamp.py
=====================
The record a trained network keeps of the database it learned from (materials/network_stamp.py): written when
a network is trained, and checked before a run. Any change to a cell of the first sheet, the 15th significant
digit of a number too, stops the run until a new network is trained, and so does a record that can't be read; a
change of colours or labels only, a conversion of units, and a network saved before networks kept a record are
used after one yes (Project.confirm_network), and a script's run refuses until then; the networks that come with
LSR MMTO match their databases and are never stamped again.

    python -m pytest tests/materials/test_network_stamp.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

DATABASE = os.path.join(ROOT, "data", "databases", "6Materials_AF_3Attr")


def _own(folder):
    """The 6-material database and its network copied into folder as mats; the .xlsx path."""
    xlsx = os.path.join(folder, "mats.xlsx")
    shutil.copy(DATABASE + ".xlsx", xlsx)
    shutil.copy(DATABASE + ".nt", os.path.join(folder, "mats.nt"))
    return xlsx


def _saved_by_the_editor(folder):
    """_own, written again as the editor saves it, and its network stamped with it as written."""
    from lsr_mmto.materials.network_stamp import stamp
    xlsx = _own(folder)
    _rewritten(xlsx, lambda *a: None)
    stamp(os.path.join(folder, "mats.nt"), xlsx)
    return xlsx


def _rewritten(xlsx, change):
    """The database written again as the editor saves it, after change(names, labels, colours, values)."""
    from lsr_mmto.materials.databases import database_table, material_display, write_database
    names, attributes, units, values = database_table(xlsx)
    colours, labels = material_display(xlsx)
    names, labels, colours, values = list(names), list(labels), list(colours), values.copy()
    change(names, labels, colours, values)
    write_database(xlsx, attributes, units, names, labels, colours, values.tolist())


def _project(xlsx):
    from lsr_mmto.problem.project import Project
    p = Project.load(os.path.join(ROOT, "projects", "U-bracket.mmto"))
    p.database = xlsx
    return p


def test_every_network_that_comes_with_lsr_mmto_matches_its_database():
    """The four included databases' networks and the thermal example's carry a record that matches their
    database."""
    import glob
    from lsr_mmto.materials.network_stamp import SAME, compare, read_network
    networks = glob.glob(os.path.join(ROOT, "data", "databases", "*.nt")) + \
        [os.path.join(ROOT, "lsr_mmto", "physics", "thermal", "example_materials.nt")]
    assert len(networks) == 5
    for nt in networks:
        assert read_network(nt)[1] is not None, nt
        assert compare(os.path.splitext(nt)[0] + ".xlsx", nt).state == SAME, nt


def test_a_trained_network_keeps_the_database_it_learned_from():
    """Training writes the database as it was read into the network's file; loading reads the weights back."""
    from lsr_mmto.materials.databases import DATABASE_VAE, VAEParams
    from lsr_mmto.materials.encoder import MaterialEncoder
    from lsr_mmto.materials.network_stamp import SAME, compare, database_stamp, read_network
    with tempfile.TemporaryDirectory() as d:
        xlsx = _own(d)
        vae = VAEParams()
        for k, v in DATABASE_VAE["6Materials_AF_3Attr"].items():
            setattr(vae, k, v)
        enc = MaterialEncoder(vae)
        enc.readExcel(xlsx)
        nt = os.path.join(d, "trained.nt")
        enc.trainAutoencoder(20, vae.klFactor, nt, vae.learningRate, 1e-9)
        weights, record = read_network(nt)
        assert record == database_stamp(xlsx)
        assert compare(xlsx, nt).state == SAME
        again = MaterialEncoder(vae)
        again.readExcel(xlsx)
        again.loadAutoencoderFromFile(nt)                                   # reads the weights out of the record
        assert all(torch.equal(again.vaeNet.state_dict()[k], weights[k]) for k in weights)


def test_the_smallest_change_of_a_number_stops_the_run():
    """Saved again unchanged: the same. The 15th significant digit of one number, or spaces around a name: no
    run, the cell named, and nothing to confirm."""
    from lsr_mmto.materials.network_stamp import CHANGED, SAME, compare
    with tempfile.TemporaryDirectory() as d:
        xlsx = _saved_by_the_editor(d)
        nt = os.path.join(d, "mats.nt")
        _rewritten(xlsx, lambda *a: None)                                   # saved again, nothing changed
        assert compare(xlsx, nt).state == SAME
        p = _project(xlsx)
        assert p.validate() == [], p.validate()

        def nudge(names, labels, colours, values):                     # the 15th significant digit, the
            v = values[1, 0]                                               # last one Excel keeps
            values[1, 0] = v + 10.0 ** (np.floor(np.log10(abs(v))) - 14)
        _rewritten(xlsx, nudge)
        check = compare(xlsx, nt)
        assert check.state == CHANGED and "cell B4" in check.text, check.text
        errors = [str(e) for e in _project(xlsx).validate()]
        assert len(errors) == 1 and "mats.xlsx: the database has changed since its network was trained" in \
            errors[0] and "Versions..." in errors[0], errors
        try:
            _project(xlsx).confirm_network()
            raise AssertionError("a changed number was confirmed")
        except ValueError as e:
            assert "a new network has to be trained" in str(e)

        _rewritten(xlsx, lambda names, *a: names.__setitem__(0, names[0] + " "))   # a name: changed too
        assert compare(xlsx, nt).state == CHANGED


def test_colours_or_labels_and_a_network_with_no_record_are_used_after_one_yes():
    """No error in the project's check, but a script's run refuses until confirm_network, which stamps the
    network (its weights unchanged); confirming a network that matches does nothing."""
    from lsr_mmto.problem.project import _ready_to_run
    from lsr_mmto.materials.network_stamp import DISPLAY, NO_STAMP, SAME, compare, read_network
    with tempfile.TemporaryDirectory() as d:
        xlsx = _saved_by_the_editor(d)
        nt = os.path.join(d, "mats.nt")
        weights = read_network(nt)[0]

        _rewritten(xlsx, lambda names, labels, colours, values: colours.__setitem__(0, "#123456"))
        check = compare(xlsx, nt)
        assert check.state == DISPLAY and "colour #123456, was #f58231" in check.text, check.text
        p = _project(xlsx)
        assert p.network_check().state == DISPLAY and p.validate() == []
        with pytest.raises(ValueError, match=r"call confirm_network\(\) on the project once"):
            _ready_to_run(p)
        p.confirm_network()
        assert compare(xlsx, nt).state == SAME and p.validate() == []
        _ready_to_run(p)
        p.confirm_network()                                                 # nothing to confirm: nothing done

        torch.save(weights, nt)                                             # as networks were saved before
        p = _project(xlsx)
        assert p.network_check().state == NO_STAMP and p.validate() == []
        with pytest.raises(ValueError, match="before networks kept a record"):
            _ready_to_run(p)
        p.confirm_network()
        assert compare(xlsx, nt).state == SAME
        assert all(torch.equal(read_network(nt)[0][k], weights[k]) for k in weights)

        torch.save({"weights": weights, "database": {"main": None}}, nt)   # a record that can't be read
        errors = [str(e) for e in _project(xlsx).validate()]
        assert len(errors) == 1 and "record of the database could not be read" in errors[0], errors


def test_a_changed_database_that_comes_with_lsr_mmto_is_not_confirmed():
    """Its network is never stamped again: the file is put back, or copied and trained as the user's own."""
    from lsr_mmto.materials import network_stamp
    from lsr_mmto.materials.network_stamp import DISPLAY, Comparison
    p = _project("6Materials_AF_3Attr")
    assert p.validate() == []
    real = network_stamp.compare
    network_stamp.compare = lambda xlsx, nt: Comparison(DISPLAY, "only colours or labels changed")
    try:
        errors = [str(e) for e in p.validate()]
        assert len(errors) == 1 and "comes with LSR MMTO, does not match its network" in errors[0], errors
        with pytest.raises(ValueError, match="comes with LSR MMTO"):
            p.confirm_network()
    finally:
        network_stamp.compare = real
    with pytest.raises(ValueError, match="comes with LSR MMTO"):
        network_stamp.stamp(DATABASE + ".nt", DATABASE + ".xlsx")
    from lsr_mmto.materials.databases import in_shipped_folder
    assert in_shipped_folder(os.path.join(ROOT, "lsr_mmto", "physics", "thermal", "example_materials.nt"))


def test_converted_units_are_asked_about_once_and_a_changed_database_stays_changed():
    """Convert to ... units on the Airplane bracket (mmKS, its database in kg/m^3, Pa and USD/kg) in a folder of
    its own: the network keeps its weights and asks once, naming the conversion. Converted after a number
    changed, the network still refuses."""
    from lsr_mmto.materials.network_stamp import CHANGED, SAME, TO_CONFIRM, read_network
    from lsr_mmto.problem.project import Project, _ready_to_run
    from lsr_mmto.problem.project_folder import convert_database_units, pack
    for change_first in (False, True):
        with tempfile.TemporaryDirectory() as d:
            p = pack(Project.load(os.path.join(ROOT, "projects", "Airplane bracket.mmto")), os.path.join(d, "A"))
            xlsx, nt = p.database_paths()[0], p.network_path()
            weights = read_network(nt)[0]
            if change_first:
                _rewritten(xlsx, lambda names, labels, colours, values: values.__setitem__((0, 0), 2 * values[0, 0]))
            assert convert_database_units(p)[0]
            check = p.network_check()
            if change_first:
                assert check.state == CHANGED, check
                continue
            assert check.state == TO_CONFIRM and check.text.startswith("the units were converted (Density kg/m^3 "
                                                                       "to t/mm^3"), check.text
            assert p.validate() == [], p.validate()
            with pytest.raises(ValueError, match="the units were converted"):
                _ready_to_run(p)
            p.confirm_network()
            assert p.network_check().state == SAME
            assert all(torch.equal(read_network(nt)[0][k], weights[k]) for k in weights)


def test_single_material_and_a_network_to_train_are_not_checked():
    """No trained network in use (single material, or Use the trained network unticked): nothing to check."""
    from lsr_mmto.materials.databases import database_table
    with tempfile.TemporaryDirectory() as d:
        xlsx = _own(d)
        _rewritten(xlsx, lambda names, labels, colours, values: values.__setitem__((0, 0), values[0, 0] * 2))
        p = _project(xlsx)
        p.material_source = "one_database_material"
        p.pinned_material = str(database_table(xlsx)[0][0])
        assert p.network_check() is None
        p.material_source = "network"
        p.pinned_material = None
        p.settings["vae"]["use_pretrained_vae"] = False
        assert p.network_check() is None
