"""
test_new_physics_in_the_gui.py
==============================
A new physics needs no GUI code: a made-up one (one field, phi; one property; one kind of support; one
objective; one setting; one result), registered as its line in available_physics.MODULES would register it,
shows in each panel that shows a physics' pieces, opened on a project of it: its kinds in Supports and loads,
its objective in Formulation, its setting in Settings, its property among a new database's columns, its
result in the Results (i); Materials, which shows nothing of a physics, opens on it.
It has nothing to pick on the part, so Click on the part is greyed, saying why. Nothing of it is written in
gui/: what the panels show comes from its entries alone.

    python -m pytest tests/gui/test_new_physics_in_the_gui.py
"""
import os
import sys
import types

import torch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests", "gui"))
sys.path.insert(0, os.path.join(ROOT, "tests", "physics"))

from PyQt5 import QtWidgets  # noqa: E402

from test_gui import Box, answers, opened, scratch  # noqa: E402
from core_lists_for_a_test import registered  # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _hold(bc, nodes, mesh, state, where):
    """The made-up support: phi held at its value on the nodes."""
    for n in nodes:
        state["values"][int(n)] = float(bc["value"])


def made_up_physics():
    """A physics no GUI code knows: every entry it must give, the optional ones left out."""
    m = types.ModuleType("made_up_module")
    m.NAME, m.DOC = "made_up", "a made-up physics, for the tests only"
    m.FIELDS = {"phi": 1}
    m.PROPERTIES = ["Phi_Conductance"]
    m.NEW_PROPERTIES = [dict(name="Phi_Conductance", measures="ratio", doc="how well phi passes",
                             shown="Phi conductance", low=0)]
    m.NEEDS = ("support",)
    m.SUPPORT_AND_LOAD_KINDS = {
        "held_phi": dict(field="phi", role="support", apply=_hold, shown="Held phi",
                         help="Holds phi at a set value.",
                         values=[dict(name="value", kind="number", label="Phi", doc="the phi held")])}
    m.QUANTITIES = [dict(name="PhiEnergy", objective=True, constraint=False, needs=["Phi_Conductance"],
                         fn=lambda model, opt: torch.zeros(()), doc="phi energy", shown="Phi energy",
                         help="How much phi the part holds.")]
    m.SETTINGS = [dict(section="advanced", key="phi_scale", default=1.0, label="Phi scale",
                       help="A setting for the tests only.", group="Material model")]
    m.RESULT_FIELDS = {"phi": dict(view="Phi", title="Phi", measures="ratio", where="node")}
    m.make_solver = lambda *args: None
    m.analyse = lambda model: None
    m.final_summary = lambda *args, **kwargs: {}
    return m


def test_a_new_physics_shows_in_every_panel_with_no_gui_code():
    from gui import database_editor, formulation_panel, materials_panel, results_panel, settings_panel
    from gui import supports_and_loads_panel as SP
    with registered(made_up_physics()), scratch() as d:
        w = opened("L-bracket", d)
        p = w.project
        p.physics, p.bcs = "made_up", []
        p.formulation = {"objective": None, "constraints": []}

        b = SP.BCDialog(w, p)                                               # Supports and loads: its kind
        assert [b.kind.itemText(i) for i in range(b.kind.count())] == ["Held phi"]
        b.close()
        panel = w.open_supports_and_loads_panel()
        menu = [x for x in panel.findChildren(QtWidgets.QPushButton) if x.text() == "Add"][0].menu()
        panel.fill_add_menu(menu)                                           # nothing to pick: greyed, saying why
        click, by_hand = menu.actions()
        assert click.text() == "Click on the part..." and not click.isEnabled() and "Enter by hand" in click.toolTip()
        assert by_hand.text() == "Enter by hand..." and by_hand.isEnabled()

        f = formulation_panel.FormulationPanel(w, p)                        # Formulation: its objective
        assert "Phi energy" in [f.objective.itemText(i) for i in range(f.objective.count())]

        assert ("advanced", "phi_scale") in settings_panel.all_setting_keys("made_up")    # Settings: its setting
        s = settings_panel.SettingsPanel(w, p)
        assert "Phi scale" in [label.text() for label in s.findChildren(QtWidgets.QLabel)]

        editor = database_editor.DatabaseEditor(w, d, project=p)            # a new database: its property
        assert "Phi_Conductance" in editor.attributes, editor.attributes
        m = materials_panel.MaterialsPanel(w, p)                            # Materials: shows no physics, opens
        r = results_panel.ResultsWindow(w)                                  # Results: its result, in the (i)
        assert "(Phi)" in r.help.toolTip(), r.help.toolTip()
        for widget in (f, s, editor, m, r):
            widget.close()
        with answers(Box.No):                                               # closed without saving
            w.close()
