"""
test_at_most_and_at_least.py
============================
A constraint is at most its limit (<=) or at least it (>=), for every quantity of every physics, with no code in
any physics: each test below runs over available_physics.MODULES, so a new physics is tested as it is added.
The optimizer sees g = value / limit - 1 or g = 1 - value / limit, both met at g <= 0; a run's record keeps g
and its op, and value / limit comes back from them. A new limit starts by its quantity's rule, 40% of its
reference at most and 10% at least, worked out in the design's material when it is one, and a limit still at
its start follows a change of materials (restarted_limits). One short U-bracket run with an at-least limit
checks the whole way, from the project to the Summary.

    python -m pytest tests/problem/test_at_most_and_at_least.py
"""
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from lsr_mmto.formulation import quantities as Q  # noqa: E402
from lsr_mmto.physics.available_physics import MODULES  # noqa: E402

UBRACKET = os.path.join(ROOT, "projects", "U-bracket.mmto")
CASES = [(m.NAME, q.name) for m in MODULES for q in Q.QUANTITIES if q.constraint and q.applies_to(m.NAME)]


def _project(physics):
    """The U-bracket project as a project of physics, with no constraints yet."""
    from lsr_mmto.problem.project import Project
    p = Project.load(UBRACKET)
    p.physics = physics
    p.formulation["constraints"] = []
    return p


def _expression(q):
    """A constraint on q as a project writes it: its required arguments as the first material (row 0)."""
    return f"{q.name}({', '.join('0' for a in q.arguments if a[2] is None)})"


@pytest.mark.parametrize("physics, name", CASES)
def test_every_quantity_of_every_physics_takes_at_most_and_at_least(physics, name):
    """Either op is accepted by the project's check and goes to the run as [name, options, limit, op]; any other
    op is refused, saying which two there are."""
    from lsr_mmto.formulation.formulation_text import to_params_form
    q = Q.find_quantity(name)
    for op in Q.OPS:
        p = _project(physics)
        p.constraint(_expression(q), op, 0.5)
        assert not [e for e in p.validate() if "operator" in str(e)], p.validate()
        spec = to_params_form(_expression(q), "constraint", 0.5) + (op,)
        assert Q.op_of(spec) == op and Q.op_of(spec[:3]) == "<="       # a record from before >=: at most
    p = _project(physics)
    p.constraint(_expression(q), "<", 0.5)
    assert any("'<=' or '>='" in str(e) for e in p.validate())


@pytest.mark.parametrize("v, b", [(0.3, 0.5), (0.7, 0.5), (-0.3, -0.5), (-0.7, -0.5)])
def test_the_optimizer_sees_g_met_at_or_below_zero_either_way(v, b):
    """The model's g at value v and limit b (the same for every quantity: it reads only the value): (v - b) / |b|
    at most and (b - v) / |b| at least, v / b - 1 and 1 - v / b for a positive limit, its gradient +1/|b| and
    -1/|b|; met exactly when the value is within its limit, a negative limit too; value / limit comes back from
    g, op and limit (share_of_limit)."""
    from lsr_mmto.optimize.design_model import MMTOModelAD
    for op, g_want, slope in (("<=", (v - b) / abs(b), 1 / abs(b)), (">=", (b - v) / abs(b), -1 / abs(b))):
        value = torch.tensor(v, dtype=torch.float64, requires_grad=True)
        model = SimpleNamespace(_value=lambda q, opt, x=value: x)
        g = MMTOModelAD._constraint(model, "Mass", None, b, op)
        (dg,) = torch.autograd.grad(g, value)
        g = float(g.detach())
        assert abs(g - g_want) < 1e-15 and abs(float(dg) - slope) < 1e-15
        assert abs(Q.share_of_limit(g, op, b) - v / b) < 1e-15
        assert (g <= 0) == (v <= b if op == "<=" else v >= b)


@pytest.mark.parametrize("physics", [m.NAME for m in MODULES])
def test_every_constraint_of_a_project_goes_to_the_run_with_its_op(physics):
    """A project of each physics with every one of its constraint quantities, at most and at least in turn:
    what Project.build gives the optimizer (formulation_params) holds each as [name, options, limit, op]."""
    p = _project(physics)
    names = [name for each, name in CASES if each == physics]
    for k, name in enumerate(names):
        p.constraint(_expression(Q.find_quantity(name)), Q.OPS[k % 2], 0.5)
    constraints = p.formulation_params()[1]
    assert [(c[0], c[2], c[3]) for c in constraints] == [(n, 0.5, Q.OPS[k % 2]) for k, n in enumerate(names)]


@pytest.mark.parametrize("physics, name", CASES)
def test_a_new_limit_starts_by_its_rule(physics, name):
    """A quantity with a reference starts at 40% of it at most and 10% at least, and says both rules in its
    (i); one with a fixed start keeps it at most and has none at least."""
    q = Q.find_quantity(name)
    p = _project(physics)
    p.database = "CustomAlloys20_3D"                    # Density and Cost: every reference can be worked out
    most, least = q.first_bound(p, "<="), q.first_bound(p, ">=")
    if q.start_reference is not None:
        assert most is not None, name
        assert abs(least - most * Q.AT_LEAST_SHARE / Q.AT_MOST_SHARE) <= 1e-12 * abs(most)
        assert "An at-most limit starts at 40% and an at-least limit at 10%." in q.start_text(p, ">=")
        assert q.start_text(p, ">=").startswith(f"Starting limit {least:.3g}")
    elif q.start_bound is not None:
        assert least is None and most is not None


def test_mass_and_cost_start_in_the_designs_material_and_follow_it():
    """Mass starts at 40% of the part solid in the design's material (the lightest of a mix), and Cost at 40%
    of it solid in the design's material (the cheapest of a mix, by density times cost); changing to a single
    material moves a limit still at its start to the new one (restarted_limits), saying so, and leaves a typed
    limit as it is."""
    from lsr_mmto.materials.databases import database_table
    from lsr_mmto.problem.project import Project
    p = Project.load(UBRACKET)
    p.database = "CustomAlloys20_3D"
    names, attributes, _, values = database_table(p.database_paths()[0])
    density, cost = values[:, attributes.index("Density")], values[:, attributes.index("Cost")]
    volume = Q.part_volume(p.stl_path())
    mass, price = Q.find_quantity("Mass"), Q.find_quantity("Cost")
    assert abs(mass.first_bound(p) - 0.4 * density.min() * volume) <= 1e-12 * mass.first_bound(p)
    assert abs(price.first_bound(p) - 0.4 * (density * cost).min() * volume) <= 1e-12 * price.first_bound(p)
    assert "its cheapest material" in price.start_text(p)
    p.formulation["constraints"] = [{"expression": "Mass()", "op": "<=", "bound": mass.first_bound(p)},
                                    {"expression": "Cost()", "op": ">=", "bound": price.first_bound(p, ">=")},
                                    {"expression": "Mass()", "op": ">=", "bound": 1.0}]       # typed
    heavy = int(np.argmax(density))
    single = p.copy(material_source="one_database_material", pinned_material=names[heavy])
    lines = Q.restarted_limits(Q.starting_limits(p), single)
    bounds = [c["bound"] for c in single.formulation["constraints"]]
    assert abs(bounds[0] - 0.4 * density[heavy] * volume) <= 1e-12 * bounds[0]
    assert abs(bounds[1] - 0.1 * density[heavy] * cost[heavy] * volume) <= 1e-12 * bounds[1]
    assert bounds[2] == 1.0 and len(lines) == 2
    fixed = p.copy(pinned_material=names[heavy])                          # Multi material, every element fixed
    assert Q.design_material(fixed) == heavy and Q.design_material(p) is None
    from lsr_mmto.materials.databases import material_display
    label = material_display(p.database_paths()[0])[1][heavy]               # as the app shows the material
    assert f"its material, {label}" in lines[0] and "Starting limit of Cost >= now" in lines[1]


def test_a_run_with_an_at_least_limit_is_recorded_and_shown_as_one():
    """A short U-bracket run with Mass at least its 10% start: the record keeps the op, value / limit comes back
    from the recorded g, the Summary shows "at least" and no Used for it, met agrees with value >= limit, Compare
    shows met alone, and the run's lines print ">="."""
    import contextlib
    import io
    from lsr_mmto.problem.project import Project, run
    from lsr_mmto.results import compare
    from lsr_mmto.results.load_results import load
    from lsr_mmto.results.tables import MET_TOLERANCE, Table, summary
    p = Project.load(UBRACKET)
    p.n_dof = 3000
    p.settings["common"]["maxIterations"] = 2
    least = Q.find_quantity("Mass").first_bound(p, ">=")
    p.constraint("Mass()", ">=", least)
    out = io.StringIO()
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(out):
        r = load(run(p, save_dir=d, verbose=True))
    i = len(r.constraint_spec) - 1
    assert r.constraint_op(i) == ">=" and r.constraint_spec[i][3] == ">="
    mass = Q.share_of_limit(r.constraint(i), ">=") * least
    assert abs(r.utilisation(i) * least - mass) <= 1e-12 * least
    assert (r.constraint(i) <= MET_TOLERANCE) == (mass >= least * (1 - MET_TOLERANCE))
    rows = [row for t in summary(r) if isinstance(t, Table) for row in t.rows if str(row[0]) == "Mass"]
    assert rows[-1][2].startswith("at least ") and rows[-1][3] == ""     # the U-bracket's own Mass limit first
    both = [row for t in compare.table({"a": r, "b": r}) if isinstance(t, Table) for row in t.rows
            if str(row[0]) == "Mass"]
    assert "%" not in str(both[-1][1])
    assert f"(Mass): " in out.getvalue() and f">= {least:.3g}?" in out.getvalue()
