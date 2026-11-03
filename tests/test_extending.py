"""
test_extending.py
=================
A new setting is one entry of SETTINGS (lsr_mmto/problem/settings.py), and a new objective or
constraint is one entry of QUANTITIES (lsr_mmto/physics/structural.py): nothing else is written by
hand. One made-up setting and one made-up quantity are added here, and each must reach every place
that uses it: the defaults, the settings page, the app's panel, the settings a run gets.

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
        import make_project_file_reference
        assert "| `my_knob` | 0.25 | What my knob does. |" in make_project_file_reference.generate()

        p = Project.load(os.path.join(ROOT, "projects", "U-bracket.mmto"))
        p.complete_settings()
        assert p.settings["advanced"]["my_knob"] == 0.25 and not p.validate()
        with tempfile.TemporaryDirectory() as d:
            assert _run_arguments(p, save_dir=d, verbose=False)["s"].my_knob == 0.25

        from gui import panels
        w = panels.SettingsPanel(None, p)
        row, label, _ = w.rows[("advanced", "my_knob")]
        assert label.text() == "My knob" and row.get() == 0.25
        assert ("advanced", "my_knob") in panels.all_setting_keys()
        w.close()
    assert "my_knob" not in settings.DEFAULT_SETTINGS["advanced"]


def test_one_entry_makes_a_quantity():
    from lsr_mmto.formulation import quantities
    q = quantities.Quantity("MyQuantity", None, ("VOLUME_FRACTION", "mine"), [quantities.STRUCTURAL], [],
                            lambda m, opt: m.x_phys.mean(), "mean of x_phys", "My quantity",
                            "What my quantity is.")
    with one_more(quantities.QUANTITIES, q, lambda: None):
        from gui import panels
        p = Project.load(os.path.join(ROOT, "projects", "U-bracket.mmto"))
        f = panels.FormulationPanel(None, p)
        model = f._quantity_combo("constraint").model()
        items = {model.item(i).data(0x0100): model.item(i) for i in range(model.rowCount())}
        assert items["MyQuantity"].text() == "My quantity"
        assert items["MyQuantity"].toolTip() == "What my quantity is."
        objectives = f._quantity_combo("objective").model()
        assert "MyQuantity" not in {objectives.item(i).data(0x0100) for i in range(objectives.rowCount())}
        f.close()


if __name__ == "__main__":
    test_one_entry_makes_a_setting()
    test_one_entry_makes_a_quantity()
    print("ok")
