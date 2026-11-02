"""
results.py
==========
The Results strip: the project's runs, newest first. A run clicked is drawn on the part (the
window's Design view) as Material, Density, Stress or Solid, with tabs for its convergence,
its latent space and its summary. Open other results... shows any run folder, also another
project's, for looking only. Two runs ticked and Compare: their summaries side by side, their
convergence overlaid and the two designs side by side on the 3D view, each on its own mesh.

Each run folder holds what showing it needs: the run's record and a copy of the project that
ran. Nothing here changes the project open in the window.
"""
import os

from PyQt5 import QtCore, QtWidgets

from . import design
from .panels import GREY, Info
from .run import ask_delete_run, open_folder, result_tabs, table_view

RESULTS_HELP = ("Each run of the project is listed here, newest first. Click one to see its design on the part "
                "and its convergence, latent space and summary in the tabs. Show chooses what the part shows: "
                "each element's material, its density, its stress, or only which elements are solid.\n\n"
                "Tick two runs and press Compare to see them side by side. Open other results... shows a run "
                "from anywhere, another project's too; it changes nothing in your project.")
OTHER = "Other: "
RUN_COLOURS = ("#2b3a67", "#b2452e")                     # the left run, the right run


def convergence_overlaid(runs):
    """Two runs' objective, and each constraint as a fraction of its budget, overlaid: one colour per
    run, one line style per constraint."""
    import numpy as np
    from matplotlib.figure import Figure
    fig = Figure(figsize=(5, 5.2), tight_layout=True)
    top, bottom = fig.add_subplot(211), fig.add_subplot(212)
    styles = ("-", "--", ":", "-.")
    for (name, run), colour in zip(runs.items(), RUN_COLOURS):
        J = np.asarray(run.history.get("J_phys", []), dtype=float).ravel()
        top.plot(np.arange(J.size), J, color=colour, lw=1.4, label=name)
        cons = np.asarray(run.history.get("constraints", []), dtype=float)
        if cons.size:
            cons = cons.reshape(cons.shape[0], -1) + 1.0                    # stored as value/budget - 1
            for i in range(cons.shape[1]):
                bottom.plot(np.arange(cons.shape[0]), cons[:, i], styles[i % len(styles)], color=colour,
                            lw=1.2, label=f"{name}, constraint {i + 1}")
    top.set_ylabel("objective")
    bottom.set_ylabel("constraint / budget")
    bottom.axhline(1.0, color="#999999", lw=0.8, ls="--")
    bottom.set_xlabel("iteration")
    for ax in (top, bottom):
        ax.grid(True, color="#dddddd", lw=0.5)
        ax.legend(fontsize=7)
    return fig


class ResultsWindow(QtWidgets.QDialog):
    """See the module docstring."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Results")
        self.records, self.others, self.current = {}, [], None    # folder -> Run; other folders; shown
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
        self.empty = QtWidgets.QLabel("No runs yet: Run > Start makes one.")
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
        self.field.addItems(design.FIELDS)
        self.field.currentTextChanged.connect(lambda _: self.redraw())
        show.addWidget(self.field)
        show.addStretch()
        show.addWidget(self.window.view_button())                  # Show setup / Show design
        layout.addLayout(show)
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
        layout.addLayout(after)
        self.refresh()

    # ------------------------------------------------------------------ the list
    def refresh(self):
        """The list again: this project's runs, newest first, then the other results opened."""
        from project_folder import is_run, run_facts, runs  # type: ignore
        keep = self.current
        self.others = [f for f in self.others if is_run(f)]
        self.list.blockSignals(True)
        self.list.clear()
        for folder in runs(self.window.project) + self.others:
            when, iterations, stopped = run_facts(folder)
            text = when + (f" · {iterations} iterations" if iterations is not None else "") + \
                (" · stopped" if stopped else "")
            if folder in self.others:
                text = OTHER + f"{os.path.basename(os.path.dirname(folder))}, {text}"
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.UserRole, folder)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            item.setToolTip(folder)
            self.list.addItem(item)
        self.list.blockSignals(False)
        self.empty.setVisible(self.list.count() == 0)
        if keep is not None and not self.select(keep):
            self.current = None
            self.tabs.clear()
        self._update_buttons()

    def folders(self):
        return [self.list.item(i).data(QtCore.Qt.UserRole) for i in range(self.list.count())]

    def ticked(self):
        return [self.list.item(i).data(QtCore.Qt.UserRole) for i in range(self.list.count())
                if self.list.item(i).checkState() == QtCore.Qt.Checked]

    def select(self, folder):
        """The run in folder shown; False when it is not in the list."""
        folder = os.path.abspath(folder)
        for i, f in enumerate(self.folders()):
            if os.path.normcase(os.path.abspath(f)) == os.path.normcase(folder):
                if self.list.currentRow() == i and self.current == f:
                    self.show_run(self.list.item(i))
                self.list.setCurrentRow(i)
                return True
        return False

    def _update_buttons(self):
        mine = self.current is not None and self.current not in self.others
        self.delete_button.setEnabled(mine)                         # others are for looking only
        self.folder_button.setEnabled(self.current is not None)
        self.compare_button.setEnabled(len(self.ticked()) == 2)

    def open_other(self):
        """Any run folder, also another project's, shown here for looking only."""
        from .pyto_ad import FILE_DIALOG_OPTIONS
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "A run's folder (it holds run_data.npz)",
                                                            os.path.dirname(self.current or self.window._folder()),
                                                            options=FILE_DIALOG_OPTIONS)
        if not folder:
            return None
        return self.add_other(folder)

    def add_other(self, folder):
        from project_folder import is_run  # type: ignore
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
        from utils import records  # type: ignore
        if folder not in self.records:
            self.records[folder] = records.load(os.path.join(folder, "run_data"))
        return self.records[folder]

    def _loaded(self, folder):
        """(run, mesh) of a run folder, or None after saying why not."""
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            QtWidgets.QApplication.processEvents()
            run = self.record(folder)
            return run, design.run_mesh(self.window, folder, run)
        except Exception as e:                                      # noqa: BLE001
            self.status.setText(f"<span style='color:#b00000'>This run cannot be shown: {e}</span>")
            return None
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def show_run(self, item):
        if item is None:
            return
        folder = item.data(QtCore.Qt.UserRole)
        self.current, self.comparing = folder, None
        self._update_buttons()
        self.status.setText("")
        loaded = self._loaded(folder)
        self.tabs.clear()
        if loaded is None:
            return
        run, _ = loaded
        result_tabs(self.tabs, run)
        self.redraw()

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
        self.window.set_design(lambda w: design.record(w, run, mesh, field))

    def delete_run(self):
        if self.current is None or self.current in self.others:
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
        """The two runs ticked: summaries side by side, convergence overlaid, designs side by side."""
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from utils import compare  # type: ignore
        folders = self.ticked()
        if len(folders) != 2:
            return False
        loaded = [self._loaded(f) for f in folders]
        if any(x is None for x in loaded):
            return False
        names = [self._name(f) for f in folders]
        runs = {n: run for n, (run, _) in zip(names, loaded)}
        self.comparing = list(zip(names, loaded))
        self.current = None
        self.list.blockSignals(True)
        self.list.setCurrentRow(-1)
        self.list.blockSignals(False)
        self._update_buttons()
        self.tabs.clear()
        try:
            self.tabs.addTab(FigureCanvasQTAgg(convergence_overlaid(runs)), "Convergence")
        except Exception as e:                                      # noqa: BLE001
            self.tabs.addTab(QtWidgets.QLabel(f"Could not draw: {e}"), "Convergence")
        try:
            text = str(compare.table(runs))
        except Exception as e:                                      # noqa: BLE001
            text = f"Could not compare: {e}"
        self.tabs.addTab(table_view(text), "Summary")
        self.status.setText(f"Comparing {names[0]} (left) and {names[1]} (right). Click a run to leave the "
                            f"comparison.")
        self._draw_compare()
        return True

    def _name(self, folder):
        from project_folder import run_facts  # type: ignore
        when = run_facts(folder)[0]
        return (OTHER + when) if folder in self.others else when

    def _draw_compare(self):
        field = self.field.currentText()
        (_, (run1, mesh1)), (_, (run2, mesh2)) = self.comparing
        clim = design.field_range([run1, run2], field)
        gap = design.width(mesh1) * 1.1

        def draw(w):
            design.record(w, run1, mesh1, field, clim=clim, name="design")
            design.record(w, run2, mesh2, field, shift=gap, clim=clim, name="design_2")
        self.window.set_design(draw, wide=True)
