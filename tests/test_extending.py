"""
test_extending.py
=================
A new setting is one entry of SETTINGS (lsr_mmto/problem/settings.py), and a new objective or
constraint is one entry of QUANTITIES (lsr_mmto/formulation/quantities.py): nothing else is written by
hand. One made-up setting and one made-up quantity are added here, and each must reach every place
that uses it: the defaults, the settings page, the app's panel, the settings a run gets. A physics may
leave out its optional entries, its Summary rows reach the Summary, and a new database starts with the
columns its project's physics reads.

    python tests/test_extending.py        (or: python -m pytest tests)
"""
import contextlib
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PyQt5 import QtWidgets  # noqa: E402

from lsr_mmto.problem import settings  # noqa: E402
from lsr_mmto.problem.project import Project, _run_arguments  # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)


@contextlib.contextmanager
def one_more(entries, entry, refresh):
    """entries with entry added while the block runs, refresh() after each change."""
    entries.append(entry)
    refresh()
    try:
        yield entry
    finally:
        entries.remove(entry)
        refresh()


def test_one_entry_makes_a_setting():
    knob = settings.Setting("advanced", "my_knob", 0.25, "My knob", "What my knob does.", "Run length")

    with one_more(settings.SETTINGS, knob, settings._make_views):
        assert settings.DEFAULT_SETTINGS["advanced"]["my_knob"] == 0.25
        assert "my_knob" in settings.RUN_KEYS
        sys.path.insert(0, os.path.join(ROOT, "docs"))
        import make_reference
        assert "| `my_knob` |  | 0.25 | What my knob does. |" in make_reference.generate()   # no physics: every project

        p = Project.load(os.path.join(ROOT, "projects", "U-bracket.mmto"))
        p.complete_settings()
        assert p.settings["advanced"]["my_knob"] == 0.25 and not p.validate()
        with tempfile.TemporaryDirectory() as d:
            assert _run_arguments(p, save_dir=d, verbose=False)["s"].my_knob == 0.25

        from gui import settings_panel
        w = settings_panel.SettingsPanel(None, p)
        row, label, _ = w.rows[("advanced", "my_knob")]
        assert label.text() == "My knob" and row.get() == 0.25
        assert ("advanced", "my_knob") in settings_panel.all_setting_keys()
        w.close()
    assert "my_knob" not in settings.DEFAULT_SETTINGS["advanced"]


def test_one_entry_makes_a_quantity():
    from lsr_mmto.formulation import quantities
    q = quantities.Quantity("MyQuantity", False, True, [], [],
                            lambda m, opt: m.x_phys.mean(), "mean of x_phys", "My quantity",
                            "What my quantity is.")
    with one_more(quantities.QUANTITIES, q, lambda: None):
        from gui import formulation_panel
        p = Project.load(os.path.join(ROOT, "projects", "U-bracket.mmto"))
        f = formulation_panel.FormulationPanel(None, p)
        model = f._quantity_combo("constraint").model()
        items = {model.item(i).data(0x0100): model.item(i) for i in range(model.rowCount())}
        assert items["MyQuantity"].text() == "My quantity"
        assert items["MyQuantity"].toolTip() == "What my quantity is."
        objectives = f._quantity_combo("objective").model()
        assert "MyQuantity" not in {objectives.item(i).data(0x0100) for i in range(objectives.rowCount())}
        f.close()



def test_every_quantity_has_its_gradient_checked():
    """Every quantity of the registry is in the hand list of tests/test_ad_quantities.py, which checks its
    gradient against finite differences (that file is slow; this check is not), or named there as not checked,
    with the reason. A new quantity left out of the list fails here, by its name."""
    import importlib.util
    from lsr_mmto.formulation.quantities import QUANTITIES
    spec = importlib.util.spec_from_file_location("ad_quantities", os.path.join(ROOT, "tests", "test_ad_quantities.py"))
    ad = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ad)
    checked = {(objective or constraint)[0] for _, objective, constraint in ad.QUANTITIES}
    missing = [q.name for q in QUANTITIES if q.name not in checked and q.name not in ad.NOT_CHECKED]
    assert not missing, f"no gradient check in tests/test_ad_quantities.py for {', '.join(missing)}"
    assert set(ad.NOT_CHECKED) <= {q.name for q in QUANTITIES}

def test_a_physics_may_leave_out_its_optional_entries():
    """_checked gives a physics that leaves out an OPTIONAL_ENTRIES entry its "none" value, and refuses one
    that leaves out any other entry, naming it."""
    import types

    from lsr_mmto.physics import available_physics as ap
    from lsr_mmto.physics.structural import structural_module
    required = [k for k in ap.MODULE_ENTRIES if k not in ap.OPTIONAL_ENTRIES]
    bare = types.ModuleType("bare")
    for k in required:
        setattr(bare, k, getattr(structural_module, k))
    ap._checked(bare)
    assert bare.OPTIONAL_PROPERTIES == [] and bare.SUMMARY_ROWS == {} and bare.deflation_space is None
    assert bare.supports_and_loads_error(None, None) is None
    assert bare.iteration_diagnostics({}, 0) is None
    del bare.analyse
    try:
        ap._checked(bare)
        raise AssertionError("a physics with no analyse was taken")
    except TypeError as e:
        assert "analyse" in str(e)


def test_a_physics_summary_rows_reach_the_summary():
    """Each SUMMARY_ROWS entry is a row of the Summary with the last value its history key recorded, and a
    key the run did not record (or recorded as nan) gives no row."""
    from types import SimpleNamespace

    from lsr_mmto.results.tables import physics_rows
    history = {"max_failure_factor": [2.0, 0.5]}
    run = SimpleNamespace(physics="structural", config={},
                          last=lambda k: history[k][-1] if k in history else float("nan"))
    assert physics_rows(run) == [("Max failure factor", "0.500", "")]
    history.clear()
    assert physics_rows(run) == []
    assert physics_rows(SimpleNamespace(physics="no such physics", config={}, last=None)) == []


def test_a_new_database_starts_with_the_columns_its_physics_reads():
    """Density, the properties the project needs, then its physics' optional ones, in the property list's
    order: an L-bracket that limits no stress needs no yield strength."""
    p = Project.load(os.path.join(ROOT, "projects", "L-bracket.mmto"))
    assert p.starting_properties() == ["Density", "Youngs_Modulus", "Poissons_Ratio"]
    q = Project.load(os.path.join(ROOT, "projects", "U-bracket.mmto"))
    assert q.starting_properties() == ["Density", "Youngs_Modulus", "Yield_Strength", "Poissons_Ratio"]


if __name__ == "__main__":
    test_one_entry_makes_a_setting()
    test_one_entry_makes_a_quantity()
    test_a_physics_may_leave_out_its_optional_entries()
    test_a_physics_summary_rows_reach_the_summary()
    test_a_new_database_starts_with_the_columns_its_physics_reads()
    print("ok")
