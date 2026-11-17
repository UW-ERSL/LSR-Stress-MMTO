"""
test_extending.py
=================
A new setting is one entry of CORE_SETTINGS (lsr_mmto/problem/settings.py), and a new objective or
constraint is one entry of CORE_QUANTITIES (lsr_mmto/formulation/quantities.py): nothing else is written by
hand. One made-up setting and one made-up quantity are added here, and each must reach every place
that uses it: the defaults, the settings page, the app's panel, the settings a run gets. A physics may
leave out its optional entries, its Summary rows reach the Summary, and a new database starts with the
columns its project's physics reads. Two things of one name are refused at start-up, naming both places, and
registering a physics in a test makes the core's lists as start-up does.

    python tests/code_rules/test_extending.py        (or: python -m pytest tests)
"""
import contextlib
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from PyQt5 import QtWidgets  # noqa: E402

from lsr_mmto.problem import settings  # noqa: E402
from lsr_mmto.problem.project import Project, _run_arguments  # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)


@contextlib.contextmanager
def one_more(entries, entry, refresh):
    """entries with entry added while the block runs, refresh() after each change."""
    entries.append(entry)
    try:
        refresh()
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
    """Every quantity of the registry is in the hand list of tests/gradients/test_ad_quantities.py, which checks its
    gradient against finite differences (that file is slow; this check is not), or named there as not checked,
    with the reason. A new quantity left out of the list fails here, by its name."""
    import importlib.util
    from lsr_mmto.formulation.quantities import QUANTITIES
    spec = importlib.util.spec_from_file_location("ad_quantities", os.path.join(ROOT, "tests", "gradients", "test_ad_quantities.py"))
    ad = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ad)
    checked = {(objective or constraint)[0] for _, objective, constraint in ad.QUANTITIES}
    missing = [q.name for q in QUANTITIES if q.name not in checked and q.name not in ad.NOT_CHECKED]
    assert not missing, f"no gradient check in tests/gradients/test_ad_quantities.py for {', '.join(missing)}"
    assert set(ad.NOT_CHECKED) <= {q.name for q in QUANTITIES}

def test_a_physics_may_leave_out_its_optional_entries():
    """_checked gives a physics that leaves out an OPTIONAL_ENTRIES entry its "none" value, and refuses one
    that leaves out any other entry, naming it, or gives SIMP_PENALTIES without set_simp_penalties."""
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
    half = types.ModuleType("half")                         # the SIMP entries: both or neither
    for k in required:
        setattr(half, k, getattr(structural_module, k))
    half.SIMP_PENALTIES = {"structural": (3.0, 1.5, 6.0)}
    try:
        ap._checked(half)
        raise AssertionError("SIMP_PENALTIES without set_simp_penalties was taken")
    except TypeError as e:
        assert "give both, or neither" in str(e)
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


def _like_structural(name, **entries):
    """A made-up physics named name with structural's entries but no settings of its own (they would share
    structural's names), and entries in their place."""
    import types

    from lsr_mmto.physics import available_physics as ap
    from lsr_mmto.physics.structural import structural_module
    module = types.ModuleType(name)
    for k in ap.MODULE_ENTRIES:
        if hasattr(structural_module, k):
            setattr(module, k, getattr(structural_module, k))
    module.NAME, module.SETTINGS = name, []
    for k, v in entries.items():
        setattr(module, k, v)
    return module


def _refused(module, says):
    """Registering module is refused with a message containing says, and every core list is left as it was."""
    sys.path.insert(0, os.path.join(ROOT, "tests", "physics"))
    from core_lists_for_a_test import registered

    from lsr_mmto.design.latent_penalties import PENALTIES
    from lsr_mmto.formulation.quantities import QUANTITIES
    from lsr_mmto.materials.databases import _BY_ALIAS, ATTRIBUTES
    before = ([q.name for q in QUANTITIES], [p.name for p in PENALTIES], [e.key for e in settings.SETTINGS],
              list(ATTRIBUTES), dict(_BY_ALIAS))
    try:
        with registered(module):
            pass
        raise AssertionError(f"{module.NAME} was registered; expected a refusal saying {says!r}")
    except (TypeError, ValueError) as e:
        assert says in str(e), str(e)
    assert before == ([q.name for q in QUANTITIES], [p.name for p in PENALTIES], [e.key for e in settings.SETTINGS],
                      list(ATTRIBUTES), dict(_BY_ALIAS))


def test_two_things_of_one_name_are_refused():
    """Start-up refuses, naming both places: a physics setting with a name the core already uses (it would
    take the core setting's place in every project), two different penalties of one name, two penalties that
    write one history key, two quantities of one name in one physics, a material property's alias that is
    another property's name, and a result shown as a view every run has (Material, Density or Solid)."""
    from lsr_mmto.physics.structural import structural_module
    knob = dict(section="common", key="maxIterations", default=5, label="Steps", help="Made up.", group="Main")
    _refused(_like_structural("clash", SETTINGS=[knob]),
             "two settings have the key 'maxIterations': one in the core, one in the clash physics; give the one in "
             "the clash physics another key")
    shove = dict(name="DistancePenalty", shown="Shove", help="Made up.", fn=lambda m: 0.0,
                 weight=lambda state, it: 0.0, scale="raw", key="shove", reads_latent_points=False)
    _refused(_like_structural("clash", PENALTIES=[shove]),
             "two different penalties are named 'DistancePenalty': one in the core, one in the clash physics; "
             "rename the one in the clash physics")
    _refused(_like_structural("clash", PENALTIES=[dict(shove, name="Shove", key="Pn")]),
             "the penalties 'DistancePenalty' and 'Shove' both record 'Pn'")
    twice = list(structural_module.QUANTITIES) + [structural_module.QUANTITIES[0]]
    _refused(_like_structural("clash", QUANTITIES=twice), "gives two quantities named")
    weight = dict(structural_module.NEW_PROPERTIES[0], name="Made_Up_Weight", aliases=("density",))
    _refused(_like_structural("clash", NEW_PROPERTIES=list(structural_module.NEW_PROPERTIES) + [weight]),
             "the material properties 'Density' and 'Made_Up_Weight' are both called 'density'")
    shown_as_density = {"weight": dict(view="Density", title="Weight", measures=None)}
    _refused(_like_structural("clash", RESULT_FIELDS=shown_as_density), "a view every run already has")


def test_the_reserved_views_are_the_results_panels_own():
    """RESERVED_VIEWS, the views a physics' result may not take, are exactly the views the Results panel gives
    every run (gui/draw_design.py, views with no run's results)."""
    from gui import draw_design
    from lsr_mmto.physics.available_physics import RESERVED_VIEWS
    assert set(draw_design.views()) == set(RESERVED_VIEWS)


def test_registering_in_a_test_makes_the_lists_as_start_up_does():
    """remake_lists (tests/physics/core_lists_for_a_test.py), which registered uses, makes from MODULES exactly
    the lists start-up made: the same physics, quantities, penalties, settings, properties, aliases and kinds,
    in the same order. A step start-up takes that it does not would show here."""
    sys.path.insert(0, os.path.join(ROOT, "tests", "physics"))
    from core_lists_for_a_test import remake_lists

    from lsr_mmto.design import latent_penalties
    from lsr_mmto.formulation import quantities
    from lsr_mmto.materials import databases
    from lsr_mmto.physics import available_physics
    from lsr_mmto.problem import boundary_conditions

    def lists():
        return (list(available_physics.PHYSICS), [(q.name, sorted(q.physics)) for q in quantities.QUANTITIES],
                [(p.name, sorted(p.physics)) for p in latent_penalties.PENALTIES],
                [(e.key, e.physics) for e in settings.SETTINGS], list(databases.ATTRIBUTES),
                dict(databases.SHOWN), dict(databases._BY_ALIAS), list(boundary_conditions.BC_KINDS),
                dict(settings.DEFAULT_SETTINGS), settings.RUN_KEYS)
    at_start = lists()
    remake_lists()
    assert lists() == at_start


if __name__ == "__main__":
    test_one_entry_makes_a_setting()
    test_one_entry_makes_a_quantity()
    test_a_physics_may_leave_out_its_optional_entries()
    test_a_physics_summary_rows_reach_the_summary()
    test_a_new_database_starts_with_the_columns_its_physics_reads()
    test_two_things_of_one_name_are_refused()
    test_the_reserved_views_are_the_results_panels_own()
    test_registering_in_a_test_makes_the_lists_as_start_up_does()
    print("ok")
