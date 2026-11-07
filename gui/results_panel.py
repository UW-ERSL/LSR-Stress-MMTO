"""
results_panel.py
================
The Results strip: the project's runs, newest first. A run clicked is drawn on the part (the
window's Design view) as Material, Density, Stress, Displacement (with the part shown: magnitude, x,
y or z, and the deformed shape) or Solid, with tabs for its convergence,
its latent space and its summary. Open other results... shows any run folder, also another
project's, for looking only. Two runs ticked and Compare: a Compare window of its own with their
convergence overlaid, their latent spaces side by side and their summaries in columns, and the two
designs side by side on the 3D view, each on its own mesh.

Each run folder holds what showing it needs: the run's record and a copy of the project that
ran, and the STL files once Export STL has written them (Show: Smooth (exported) draws those). Nothing
here changes the project open in the window.

An analysis (Analyse, beside Run) is listed among the runs as "Analysis: <material>"; one not saved is
listed first, until the next one, and has no folder. An analysis has no convergence and no latent
points: those tabs are left out, and a comparison with one says so in their place.
"""
import os

from PyQt5 import QtCore, QtWidgets

from . import draw_design
from .panel_widgets import GREY, Info
from .run_panel import ask_delete_run, open_folder, plot_canvas, result_tabs, summary_view, table_view

RESULTS_HELP = ("Each run of the project is listed here, newest first. Click one to see its design on the part "
                "and its convergence, latent space and summary in the tabs.\n\nShow chooses what the part shows: "
                "each element's material, its density, its stress, how far it moves (Displacement), only which "
                "elements are solid, or the smooth STL files Export STL wrote for the run.\n\n"
                "Tick two runs and press Compare to see them side by side. Open other results... shows a run "
                "from anywhere, another project's too; it changes nothing in your project.\n\n"
                "An analysis (Analyse, at the top) is listed as Analysis: and its material. One not saved is "
                "listed first until the next analysis, and is gone once Results or the project is closed.")
OTHER = "Other: "
STL_HELP = ("Export STL, at the top, writes the design shown as two kinds of smooth STL into the run's folder.\n\n"
            "design.stl is the whole part, for printing in one material. Faces the optimization did not touch keep "
            "their exact shape from your STL.\n\n"
            "design_<material>.stl is one file per material the design uses. Together they make up the part, with "
            "no gaps and no overlaps between them.\n\n"
            "Smoothing removes the staircase of the elements, but it cannot add detail finer than the mesh: "
            "features one element across are dropped and square corners round off. For a part to print, run on a "
            "finer mesh. The check written after each export says whether every file is closed and how its "
            "volume compares with the design.\n\n"
            "Show > Smooth (exported) draws the files of a run that has been exported, each piece in its "
            "material's colour. Pull the material pieces apart sets them side by side, so each can be seen alone.")
VECTOR_HELP = ("magnitude shows the size at each point, for Displacement how far it moves. x, y and z show the "
               "part along each axis, which can be below zero. These values belong to the corners of the elements, "
               "so the colours blend smoothly over the faces.\n\n"
               "Deformed shape draws the part moved by its displacement times Scale. The first Scale makes the "
               "largest movement about 5% of the part's largest size, so that the shape can be seen; the real "
               "movement is usually far smaller.")
SMOOTH = "Smooth (exported)"                             # the Show choice of a run's exported STL files
UNSAVED = "analysis not saved"                           # the list's key of the analysis shown but not saved
NO_HISTORY = "An analysis is one solve of the solid part, not an optimization: it has no {}."
RUN_COLOURS = ("#2b3a67", "#b2452e")                     # the left run, the right run


def convergence_overlaid(runs):
    """Two runs' objective, and each constraint as a fraction of its budget, overlaid: one colour per
    run, one line style per constraint, one legend under both plots (it never covers the curves)."""
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    fig = Figure(figsize=(8.0, 6.4))
    top, bottom = fig.add_subplot(211), fig.add_subplot(212)
    styles = ("-", "--", ":", "-.")
    n_cons = 0
    for k, ((name, run), colour) in enumerate(zip(runs.items(), RUN_COLOURS)):
        wide = dict(lw=3.2, zorder=2, alpha=0.75) if k else dict(lw=1.4, zorder=3)   # the second underneath and
        J = np.asarray(run.history.get("J_phys", []), dtype=float).ravel()          # wider: where the two runs
        top.plot(np.arange(J.size), J, color=colour, **wide)                        # match, both stay visible
        cons = np.asarray(run.history.get("constraints", []), dtype=float)
        if cons.size:
            cons = cons.reshape(cons.shape[0], -1) + 1.0                    # stored as value/budget - 1
            n_cons = max(n_cons, cons.shape[1])
            for i in range(cons.shape[1]):
                bottom.plot(np.arange(cons.shape[0]), cons[:, i], styles[i % len(styles)], color=colour,
                            **(dict(lw=2.8, zorder=2, alpha=0.75) if k else dict(lw=1.2, zorder=3)))
    top.set_ylabel("objective")
    bottom.set_ylabel("constraint / budget")
    bottom.axhline(1.0, color="#999999", lw=0.8, ls="--")
    bottom.set_xlabel("iteration")
    for ax in (top, bottom):
        ax.grid(True, color="#dddddd", lw=0.5)
    handles = [Line2D([], [], color=c, lw=w, label=n) for n, c, w in zip(runs, RUN_COLOURS, (1.4, 3.2))]
    handles += [Line2D([], [], color="#666666", ls=styles[i % len(styles)], lw=1.2, label=constraint_name(runs, i))
                for i in range(n_cons)]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def constraint_name(runs, i):
    """Constraint i of the runs by its name (run.constraint_label), both names when the runs differ."""
    names = []
    for run in runs.values():
        try:
            label = run.constraint_label(i) if i < len(run.constraint_spec) else None
        except Exception:                                           # noqa: BLE001
            label = None
        if label and label not in names:
            names.append(label)
    return " / ".join(names) or f"constraint {i + 1}"


def same_network(run1, run2):
    """True when two runs used the same material network: their real materials sit at the same latent points."""
    import numpy as np
    a, b = np.asarray(run1.z_real, float), np.asarray(run2.z_real, float)
    return a.shape == b.shape and np.allclose(a, b, rtol=0, atol=1e-9)


def latent_side_by_side(runs, max_points=20000):
    """Each run's latent field against its real materials, side by side, drawn as the single-run latent
    plot (red circles: the design's elements; black stars: the real materials). With the same network,
    the two share their axes (and their projection, above three latent dimensions), so where the
    elements sit compares directly."""
    import numpy as np
    from matplotlib.figure import Figure
    from lsr_mmto.results import run_plots
    names, rs = list(runs), list(runs.values())
    shared = same_network(*rs)
    dims = {np.asarray(r.z_real).shape[1] for r in rs}
    three_d = dims == {3}
    fig = Figure(figsize=(10.0, 5.0))
    pca = run_plots.latent_projection(rs[0].z_real) if shared else None
    axes = []
    for k, (name, run) in enumerate(zip(names, rs)):
        zr = np.asarray(run.z_real, float)
        z = np.asarray(run.z_solid(), float).reshape(-1, zr.shape[1])
        if z.shape[0] > max_points:
            z = z[np.random.default_rng(0).choice(z.shape[0], max_points, replace=False)]
        labels = list(run.material_labels) + [str(i) for i in range(len(run.material_labels), zr.shape[0])]
        p = pca if shared else run_plots.latent_projection(zr)
        ax = run_plots.latent_axes(fig, z, zr, labels, where=(1, 2, k + 1), pca=p, small=True)
        ax.set_title(name, fontsize=10, color=RUN_COLOURS[k])
        axes.append(ax)
    if shared and not three_d:
        lo = np.min([a.get_xlim()[0] for a in axes]), np.min([a.get_ylim()[0] for a in axes])
        hi = np.max([a.get_xlim()[1] for a in axes]), np.max([a.get_ylim()[1] for a in axes])
        for a in axes:
            a.set_xlim(lo[0], hi[0])
            a.set_ylim(lo[1], hi[1])
    marks = ("Red bars: how many of the design's elements sit at each point" if dims == {1} else
             "Red circles: the design's elements")
    note = (f"{marks}; black stars: the real materials." if shared else
            "These runs used different material networks, so their latent spaces are different spaces: "
            "compare where the elements sit relative to the stars in each, not between the two plots.")
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=9, color="#555555", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


class CompareWindow(QtWidgets.QDialog):
    """Two runs compared, in a window of its own that can be resized or maximised: convergence overlaid,
    the latent spaces side by side, the summaries in columns. Closing it ends the comparison."""

    def __init__(self, parent, runs, on_close):
        from lsr_mmto.results import compare
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.Window)
        names = list(runs)
        self.setWindowTitle(f"Compare: {names[0]} and {names[1]}")
        self.resize(1100, 760)
        self.on_close = on_close
        layout = QtWidgets.QVBoxLayout(self)
        head = QtWidgets.QLabel(f"<span style='color:{RUN_COLOURS[0]}'><b>{names[0]}</b></span> (left on the part) "
                                f"and <span style='color:{RUN_COLOURS[1]}'><b>{names[1]}</b></span> (right)")
        layout.addWidget(head)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, stretch=1)
        analysis = any(run.analysed_material is not None for run in runs.values())
        for title, make, what in (("Convergence", lambda: convergence_overlaid(runs), "convergence"),
                                  ("Latent space", lambda: latent_side_by_side(runs), "latent space")):
            if analysis:
                self.tabs.addTab(QtWidgets.QLabel(NO_HISTORY.format(what)), title)
                continue
            try:
                self.tabs.addTab(plot_canvas(make()), title)
            except Exception as e:                                  # noqa: BLE001
                self.tabs.addTab(QtWidgets.QLabel(f"Could not draw: {e}"), title)
        try:
            self.tabs.addTab(summary_view(compare.table(runs)), "Summary")
        except Exception as e:                                      # noqa: BLE001
            self.tabs.addTab(table_view(f"Could not compare: {e}"), "Summary")

    def closeEvent(self, event):
        callback, self.on_close = self.on_close, None
        if callback is not None:
            callback()
        super().closeEvent(event)


class ResultsWindow(QtWidgets.QDialog):
    """See the module docstring."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Results")
        self.records, self.others, self.current = {}, [], None    # folder -> Run; other folders; shown
        self.unsaved = None                                        # (Run, mesh) of the analysis not saved
        self.comparing = None                                      # [(name, (run, mesh))] x 2 while comparing
        layout = QtWidgets.QVBoxLayout(self)

        head = QtWidgets.QHBoxLayout()
        head.addWidget(QtWidgets.QLabel("Runs of this project"))
        head.addStretch()
        head.addWidget(Info(RESULTS_HELP))
        layout.addLayout(head)
        self.list = QtWidgets.QListWidget()
        self.list.setMinimumHeight(120)
        self.list.currentItemChanged.connect(lambda item, _: self.show_run(item))
        self.list.itemChanged.connect(lambda _: self._update_buttons())
        layout.addWidget(self.list)
        self.empty = QtWidgets.QLabel("No runs yet. Press Run at the top to make one.")
        self.empty.setStyleSheet(GREY)
        layout.addWidget(self.empty)

        buttons = QtWidgets.QHBoxLayout()
        self.compare_button = QtWidgets.QPushButton("Compare")
        self.compare_button.setToolTip("Tick two runs to compare them")
        self.compare_button.clicked.connect(self.compare)
        self.other_button = QtWidgets.QPushButton("Open other results...")
        self.other_button.clicked.connect(self.open_other)
        for widget in (self.compare_button, self.other_button):
            buttons.addWidget(widget)
        buttons.addStretch()
        layout.addLayout(buttons)

        show = QtWidgets.QHBoxLayout()
        show.addWidget(QtWidgets.QLabel("Show"))
        self.field = QtWidgets.QComboBox()
        self.field.addItems(draw_design.views())
        self.field.currentTextChanged.connect(lambda _: self.redraw())
        show.addWidget(self.field)
        self.apart = QtWidgets.QCheckBox("Pull the material pieces apart")
        self.apart.setVisible(False)
        self.apart.toggled.connect(lambda _: self.redraw())
        show.addWidget(self.apart)
        self.component = QtWidgets.QComboBox()                     # a vector: its magnitude, x, y or z
        self.component.addItems(draw_design.COMPONENTS)
        self.component.currentTextChanged.connect(lambda _: self.redraw())
        self.vector_info = Info(VECTOR_HELP)
        for widget in (self.component, self.vector_info):
            widget.setVisible(False)
            show.addWidget(widget)
        show.addStretch()
        layout.addLayout(show)
        self.deform_row = QtWidgets.QWidget()                      # under Show, for a result that moves the part
        deform = QtWidgets.QHBoxLayout(self.deform_row)
        deform.setContentsMargins(0, 0, 0, 0)
        self.deformed = QtWidgets.QCheckBox("Deformed shape")
        self.deformed.toggled.connect(lambda _: self.redraw())
        self.scale_label = QtWidgets.QLabel("Scale")
        self.scale = QtWidgets.QLineEdit()
        self.scale.setFixedWidth(100)
        self.scale.editingFinished.connect(self.redraw)
        self._scaled = None                                        # what the Scale shown was worked out for
        for widget in (self.deformed, self.scale_label, self.scale):
            deform.addWidget(widget)
        deform.addStretch()
        self.deform_row.setVisible(False)
        layout.addWidget(self.deform_row)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(GREY)
        layout.addWidget(self.status)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, stretch=1)

        after = QtWidgets.QHBoxLayout()
        self.folder_button = QtWidgets.QPushButton("Open folder")
        self.folder_button.clicked.connect(lambda: open_folder(self.current))
        self.delete_button = QtWidgets.QPushButton("Delete this run")
        self.delete_button.clicked.connect(self.delete_run)
        for widget in (self.folder_button, self.delete_button):
            after.addWidget(widget)
        after.addStretch()
        after.addWidget(Info(STL_HELP))
        layout.addLayout(after)
        self.refresh()

    # ------------------------------------------------------------------ the list
    def refresh(self):
        """The list again: this project's runs, newest first, then the other results opened."""
        from lsr_mmto.problem.project_folder import analysed_material, is_run, run_facts, runs
        keep = self.current
        self.others = [f for f in self.others if is_run(f)]
        self.list.blockSignals(True)
        self.list.clear()
        for folder in ([UNSAVED] if self.unsaved else []) + runs(self.window.project) + self.others:
            if folder == UNSAVED:
                self._add_item(UNSAVED, f"Analysis: {self.unsaved[0].analysed_material} · not saved", "")
                continue
            when, iterations, stopped = run_facts(folder)
            text = when + (f" · {iterations} iterations" if iterations is not None else "") + \
                (" · stopped" if stopped else "")
            material = analysed_material(folder)
            if material is not None:
                text = f"Analysis: {material} · {when}"
            if folder in self.others:
                text = OTHER + f"{os.path.basename(os.path.dirname(folder))}, {text}"
            self._add_item(folder, text, folder)
        self.list.blockSignals(False)
        self.empty.setVisible(self.list.count() == 0)
        if keep is not None and not self.select(keep):
            self.current = None
            self.tabs.clear()
        self._update_buttons()

    def _add_item(self, folder, text, tip):
        item = QtWidgets.QListWidgetItem(text)
        item.setData(QtCore.Qt.UserRole, folder)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.Unchecked)
        item.setToolTip(tip)
        self.list.addItem(item)

    def show_analysis(self, run, mesh, folder=None):
        """A finished analysis shown as a run is: saved (in folder), from its place among the runs; not saved,
        listed first until the next analysis."""
        self.unsaved = None if folder else (run, mesh)
        self.refresh()
        return self.select(folder or UNSAVED)

    def folders(self):
        return [self.list.item(i).data(QtCore.Qt.UserRole) for i in range(self.list.count())]

    def ticked(self):
        return [self.list.item(i).data(QtCore.Qt.UserRole) for i in range(self.list.count())
                if self.list.item(i).checkState() == QtCore.Qt.Checked]

    def select(self, folder):
        """The run in folder shown; False when it is not in the list."""
        key = (lambda f: f) if folder == UNSAVED else (lambda f: os.path.normcase(os.path.abspath(f)))
        for i, f in enumerate(self.folders()):
            if key(f) == key(folder):
                if self.list.currentRow() == i and self.current == f:
                    self.show_run(self.list.item(i))
                self.list.setCurrentRow(i)
                return True
        return False

    def _update_buttons(self):
        mine = self.current is not None and self.current not in self.others and self.current != UNSAVED
        self.delete_button.setEnabled(mine)                         # others are for looking only
        self.folder_button.setEnabled(self.current is not None and self.current != UNSAVED)
        self.compare_button.setEnabled(len(self.ticked()) == 2)

    def open_other(self):
        """Any run folder, also another project's, shown here for looking only."""
        from .pyto_ad_imports import FILE_DIALOG_OPTIONS
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "A run's folder (it holds run_data.npz)",
                                                            os.path.dirname(self.current or self.window._folder()),
                                                            options=FILE_DIALOG_OPTIONS)
        if not folder:
            return None
        return self.add_other(folder)

    def add_other(self, folder):
        from lsr_mmto.problem.project_folder import is_run
        if not is_run(folder):
            QtWidgets.QMessageBox.warning(self, "Open other results", f"{folder} holds no run: a run's folder "
                                                                      f"holds run_data.npz.")
            return None
        folder = os.path.abspath(folder)
        if folder not in self.folders():
            self.others.append(folder)
            self.refresh()
        self.select(folder)
        return folder

    # ------------------------------------------------------------------ one run
    def record(self, folder):
        from lsr_mmto.results import load_results
        if folder == UNSAVED:
            return self.unsaved[0]
        if folder not in self.records:
            self.records[folder] = load_results.load(os.path.join(folder, "run_data"))
        return self.records[folder]

    def _loaded(self, folder):
        """(run, mesh) of a run folder, or None after saying why not."""
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            QtWidgets.QApplication.processEvents()
            run = self.record(folder)
            return run, (self.unsaved[1] if folder == UNSAVED else draw_design.run_mesh(self.window, run))
        except Exception as e:                                      # noqa: BLE001
            self.status.setText(f"<span style='color:#b00000'>This run cannot be shown: {e}</span>")
            return None
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def show_run(self, item):
        if item is None:
            return
        folder = item.data(QtCore.Qt.UserRole)
        self.apart.blockSignals(True)
        self.apart.setChecked(False)
        self.apart.blockSignals(False)
        self.comparing = None                                      # a run clicked: the comparison ends
        self.end_compare()
        self.current = folder
        self._update_buttons()
        self.status.setText("")
        loaded = self._loaded(folder)
        self.tabs.clear()
        if loaded is None:
            return
        run, _ = loaded
        result_tabs(self.tabs, run)
        self._set_views(run)
        self.redraw()

    def _set_views(self, *runs):
        """The Show list for the runs shown: the views their own physics saved, and for one run Smooth
        (exported), greyed while its folder holds no exported STL; the choice kept when it can still be
        made, Material otherwise."""
        from lsr_mmto.results import stl_export
        items = list(draw_design.views(*runs)) + ([SMOOTH] if len(runs) == 1 else [])
        choice = self.field.currentText()
        self.field.blockSignals(True)
        if [self.field.itemText(i) for i in range(self.field.count())] != items:
            self.field.clear()
            self.field.addItems(items)
            self.field.ensurePolished()                             # the style sheet's font, not the default
            longest = max(self.field.fontMetrics().horizontalAdvance(t) for t in items)
            self.field.setMinimumWidth(longest + 60)                # each choice read whole, past the arrow
        if len(runs) == 1:
            smooth = self.field.model().item(items.index(SMOOTH))
            done = self.current not in (None, UNSAVED) and stl_export.exported(runs[0], self.current) is not None
            smooth.setEnabled(done)
            smooth.setToolTip("" if done else "Export STL first")
            if choice == SMOOTH and not done:
                choice = "Material"
        self.field.setCurrentText(choice if choice in items else "Material")
        self.field.blockSignals(False)

    def redraw(self):
        """The run shown (or the two compared) drawn again with the Show choice."""
        if self.comparing:
            return self._draw_compare()
        if self.current is None:
            return
        loaded = self._loaded(self.current)
        if loaded is None:
            return
        run, mesh = loaded
        field = self.field.currentText()
        self.apart.setVisible(field == SMOOTH)
        mine = (self.current, run, mesh) if self.current not in self.others and self.current != UNSAVED else None
        if field == SMOOTH:
            self._vector_controls([], field)
            return self._draw_smooth(run, mine)
        component, scale = self._vector_controls([(run, mesh)], field)
        self.window.set_design(lambda w: draw_design.record(w, run, mesh, field, component=component, scale=scale),
                               run=mine)

    def _vector_controls(self, shown, field):
        """Beside Show, for a vector result: the part shown (magnitude, x, y, z) and its (i); for one that
        deforms the part (draw_design.deforms), Deformed shape and its Scale, first worked out for the runs
        shown (draw_design.deformed_scale). Returns (component, scale or None) for the drawing."""
        info = next((run.result_info[draw_design.result_of(run, field)] for run, _ in shown
                     if draw_design.result_of(run, field) is not None), {})
        vector, deforms = info.get("shape") == "vector", draw_design.deforms(info)
        for widget in (self.component, self.vector_info):
            widget.setVisible(vector)
        self.deform_row.setVisible(deforms)
        on = deforms and self.deformed.isChecked()
        self.scale_label.setVisible(on)
        self.scale.setVisible(on)
        if not on:
            return self.component.currentText(), None
        first = draw_design.deformed_scale(shown, field)
        if self._scaled != (tuple(id(run) for run, _ in shown), field):
            self._scaled = (tuple(id(run) for run, _ in shown), field)
            self.scale.setText(f"{first:.3g}")
        try:
            scale = float(self.scale.text())
        except ValueError:
            scale = first
            self.scale.setText(f"{first:.3g}")
        return self.component.currentText(), scale

    # ------------------------------------------------------------------ STL
    def exported(self, folder):
        """Export STL (top row) wrote the run's STL files into folder: when it is the run shown here, Show
        turns to Smooth (exported) and they are drawn."""
        if self.current is None or os.path.normcase(os.path.abspath(folder)) != \
                os.path.normcase(os.path.abspath(self.current)) or self.comparing:
            return
        self._set_views(self.record(self.current))
        if self.field.currentText() == SMOOTH:
            self.redraw()
        else:
            self.field.setCurrentText(SMOOTH)                       # draws them

    def _draw_smooth(self, run, mine):
        """The STL files in the run's folder (stl_export.exported), each piece in its material's colour, with
        its label in the legend; pulled apart when ticked."""
        from lsr_mmto.results import stl_export
        found = stl_export.exported(run, self.current)
        if found is None:
            return
        whole, files = found
        pieces = [(path, run.material_colors[m], run.material_labels[m]) for m, path in files]
        exported = {m for m, _ in files}
        absent = [(c, label) for m, (c, label) in enumerate(zip(run.material_colors, run.material_labels))
                  if m not in exported] if pieces else []
        if not pieces:
            pieces = [(whole, draw_design.SOLID_COLOUR, "")]
        apart = self.apart.isChecked()
        self.window.set_design(lambda w: draw_design.smooth(w, pieces, apart, absent), wide=apart, run=mine)

    def delete_run(self):
        if self.current is None or self.current in self.others or self.current == UNSAVED:
            return
        folder = self.current
        if ask_delete_run(self, folder):
            self.records.pop(folder, None)
            self.window.message_text.append(f"Deleted the run {folder}")
            self.window.show_setup_view()
            self.window.set_design(None, show=False)
            self.current = None
            self.tabs.clear()
            self.refresh()

    # ------------------------------------------------------------------ two runs
    def compare(self):
        """The two runs ticked, in a Compare window of their own; their designs side by side on the part."""
        folders = self.ticked()
        if len(folders) != 2:
            return False
        loaded = [self._loaded(f) for f in folders]
        if any(x is None for x in loaded):
            return False
        names = [self._name(f) for f in folders]
        runs = {n: run for n, (run, _) in zip(names, loaded)}
        self.end_compare()
        self.comparing = list(zip(names, loaded))
        self._set_views(*(run for run, _ in loaded))
        self.compare_window = CompareWindow(self.window, runs, self._compare_closed)
        self.compare_window.show()
        self.status.setText(f"Comparing {names[0]} (left) and {names[1]} (right). Close the Compare window or "
                            f"click a run to leave the comparison.")
        self._draw_compare()
        return True

    def end_compare(self):
        """The Compare window closed (it calls _compare_closed)."""
        w = getattr(self, "compare_window", None)
        if w is not None:
            try:
                w.close()
            except RuntimeError:
                pass
        self.compare_window = None

    def _compare_closed(self):
        if not self.comparing:
            return
        self.comparing, self.compare_window = None, None
        self.status.setText("")
        if self.current is not None:
            self.redraw()
        else:
            self.window.show_setup_view()

    def closeEvent(self, event):
        self.end_compare()
        super().closeEvent(event)

    def _name(self, folder):
        from lsr_mmto.problem.project_folder import run_facts
        from lsr_mmto.problem.project_folder import analysed_material
        if folder == UNSAVED:
            return f"Analysis: {self.unsaved[0].analysed_material}, not saved"
        when = run_facts(folder)[0]
        material = analysed_material(folder)
        when = when if material is None else f"Analysis: {material}, {when}"
        return (OTHER + when) if folder in self.others else when

    def _draw_compare(self):
        field = self.field.currentText()
        (_, (run1, mesh1)), (_, (run2, mesh2)) = self.comparing
        component, scale = self._vector_controls([(run1, mesh1), (run2, mesh2)], field)
        clim = draw_design.field_range([(run1, mesh1), (run2, mesh2)], field, component)
        gap = draw_design.width(mesh1) * 1.1

        def draw(w):
            draw_design.record(w, run1, mesh1, field, clim=clim, name="design", component=component, scale=scale)
            draw_design.record(w, run2, mesh2, field, shift=gap, clim=clim, name="design_2", component=component,
                               scale=scale)
        self.window.set_design(draw, wide=True)
