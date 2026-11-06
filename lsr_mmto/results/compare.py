"""
compare.py
==========
Saved runs side by side (the app's Compare): runs is a dict of name to Run (records.load).

settings(runs)             the run arguments side by side; differences are marked
table(runs)                final values and run times side by side
histories(runs, out=None)  objective, constraints and grayness over the iterations
designs(runs, out=None)    the optimized designs side by side, coloured by material
"""
from __future__ import annotations

import numpy as np

# arguments that say where a run was written or how it printed, not what it solved
_IGNORE = {"save_dir", "plot_progress", "verbose", "plotter", "timeLimit", "saveNet"}
_SWITCHED = {"use_z_smoothing", "use_penalization"}   # changed by the schedule during a run


def _arguments(run):
    return dict((run.config or {}).get("arguments_resolved") or {})


def settings(runs):
    """Run arguments, one column per run; a row is marked where they differ."""
    from .tables import Table
    names = list(runs)
    args = {k: _arguments(r) for k, r in runs.items()}
    keys = sorted(set().union(*[a.keys() for a in args.values()]) - _IGNORE - _SWITCHED)
    rows, differ = [], []
    for k in keys:
        vals = [str(args[n].get(k, "")) for n in names]
        same = len(set(vals)) == 1
        rows.append([k] + vals + ["" if same else "differs"])
        if not same:
            differ.append(k)
    note = ("identical settings" if not differ
            else "settings differ: " + ", ".join(differ))
    note += (". use_z_smoothing and use_penalization are left out: the run record stores "
             "them after the run, when the schedule may have switched them off.")
    return Table(rows, headers=["argument"] + names + [""],
                 title="Run settings", note=note,
                 highlight=lambda row: row[-1] == "differs", symbol_cols=())


def _runtime(run):
    return float(((run.config or {}).get("run") or {}).get("runtime_seconds") or np.nan)


def _iterations(run):
    return run.iterations                    # the starting design is not an iteration (records.Run.iterations)


def table(runs):
    """The runs side by side, one column each, in the sections of a run's summary (tables.summary):
    the result with each constraint met or not, the design, the materials and the run. A constraint an
    analysis cannot give is empty in its column, and left out when no run gives it."""
    from .tables import Heading, Table, Tables, _unit, constraint_name, given, met, met_cell
    names = list(runs)
    rs = [runs[n] for n in names]
    r0 = rs[0]
    obj = (r0.config or {}).get("topopt_params", {}).get("Objective") or ["objective", None]
    unit = _unit(r0, obj[0])
    right = tuple(range(1, len(rs) + 1))
    result = [[f"{r0.objective_label}" + (f" ({unit})" if unit else "")] + [f"{r.final_objective:.4g}" for r in rs]]
    if any(np.isfinite(r.max_ff) for r in rs):
        result.append(["Max failure factor"] + [f"{r.max_ff:.3f}" for r in rs])
    for i in range(len(r0.constraint_spec)):
        if not any(given(r, i) for r in rs):                # an analysis's constraint it cannot give: left out
            continue
        result.append([f"{constraint_name(r0, i)}, used"] +
                      [met_cell(met(r, i), f"{100 * r.utilisation(i):.1f} %") if given(r, i) else "" for r in rs])
    return Tables(
        Heading(f"{r0.example}: the runs side by side"),
        Table(result, headers=[""] + names, title="Result", right_cols=right),
        Table([["Elements"] + [f"{r.n_elems:,}" for r in rs],
               ["Solid"] + [f"{int(r.solid.sum()):,} ({100 * r.solid.mean():.1f} %)" for r in rs],
               ["Grey (in between)"] + [f"{100 * r.grey:.1f} %" if np.isfinite(r.grey) else "" for r in rs]],
              title="Design", right_cols=right),
        Table([["Used"] + [f"{int(r.materials_used().size)} of {len(r.material_names)}" for r in rs],
               ["Latent dimension"] + [str(r.n_latent) for r in rs]], title="Materials", right_cols=right),
        Table([["Iterations"] + [f"{_iterations(r)}" for r in rs],
               ["Run time"] + [f"{_runtime(r) / 60:.1f} min" for r in rs],
               ["Time per iteration"] + [f"{_runtime(r) / max(_iterations(r), 1):.2f} s" for r in rs]],
              title="Run", note="Run time is the optimization only, from the run's own record.", right_cols=right))


def histories(runs, out=None):
    """Objective, constraints (fraction of budget) and grayness, all runs overlaid,
    and the relative difference of each objective history from the first run's."""
    import matplotlib.pyplot as plt
    from .run_plots import _save

    names = list(runs)
    styles = ["-", "--", ":", "-."]
    colours = ["#2b3a67", "#b2452e", "#2a7f62", "#7a5c99"]

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    a_obj, a_con, a_grey, a_diff = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]

    J_ref = np.asarray(runs[names[0]].history.get("J_phys", []), dtype=float).ravel()
    for k, n in enumerate(names):
        h = runs[n].history
        ls, c = styles[k % len(styles)], colours[k % len(colours)]
        J = np.asarray(h.get("J_phys", []), dtype=float).ravel()
        it = np.arange(1, J.size + 1)
        a_obj.plot(it, J, ls, color=c, lw=1.4, label=n)

        cons = np.asarray(h.get("constraints", []), dtype=float)
        if cons.size:
            cons = cons.reshape(cons.shape[0], -1) + 1.0      # stored as value/budget - 1
            for i in range(cons.shape[1]):
                a_con.plot(np.arange(1, cons.shape[0] + 1), cons[:, i], ls, color=c, lw=1.2,
                           label=f"{n}, {runs[n].constraint_label(i)}")

        g = np.asarray(h.get("grey", []), dtype=float).ravel()
        a_grey.plot(np.arange(1, g.size + 1), 100 * g, ls, color=c, lw=1.4, label=n)

        if k > 0 and J_ref.size:
            m = min(J.size, J_ref.size)
            d = np.abs(J[:m] - J_ref[:m]) / np.maximum(np.abs(J_ref[:m]), 1e-300)
            a_diff.semilogy(np.arange(1, m + 1), np.maximum(d, 1e-16), ls, color=c, lw=1.2,
                            label=f"{n} vs {names[0]}")

    a_obj.set_title(f"objective: {runs[names[0]].objective_label}")
    a_con.set_title("constraints, fraction of budget")
    a_con.axhline(1.0, color="#999999", lw=1.0, ls="--")
    a_grey.set_title("grayness (%)")
    a_diff.set_title("relative difference in the objective")
    for a in (a_obj, a_con, a_grey, a_diff):
        a.set_xlabel("iteration")
        a.grid(True, color="#cccccc", lw=0.5, alpha=0.6)
        a.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out)


def designs(runs, out=None, *, size=(1000, 800), view="iso"):
    """The optimized designs side by side, coloured by material, one shared legend.

    The same raw drawing as run_plots.design: one quad per element in 2D, one voxel
    face per element in 3D. No smoothing.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from .run_plots import _panel_image, _save, is_3d
    from .design_3d import draw_panel, mesh_for

    names = list(runs)
    rs = [runs[n] for n in names]
    first = rs[0]
    mesh = mesh_for(first)

    if is_3d(first):
        fig, axs = plt.subplots(1, len(rs), figsize=(4.6 * len(rs), 4.2))
    else:
        fig, axs = plt.subplots(1, len(rs), figsize=(4.4 * len(rs), 4.0))
    axs = np.atleast_1d(axs)

    for ax, n, r in zip(axs, names, rs):
        if is_3d(r):
            ax.imshow(_panel_image(r, mesh, None, size, None, view=view))
            ax.axis("off")
        else:
            draw_panel(ax, r, "material", mesh=mesh)
        ax.set_title(n)

    labels, colours = first.material_labels, first.material_colors
    used = sorted(set().union(*[set(r.materials_used().tolist()) for r in rs]))
    handles = [Patch(fc=colours[i], ec="0.3", lw=0.5,
                     label=labels[i] if i < len(labels) else f"material {i}") for i in used]
    if handles:
        axs[-1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                       frameon=False, fontsize=10)
    fig.tight_layout()
    return _save(fig, out)
