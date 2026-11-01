"""
run.py
======
The Run strip: an optimization of the project, started from the window.

RunWindow is PyTO-AD's Optimize window (StructuralTopOptWindow): its worker thread, its Qt
signals, its live plot, its live redraw of the design and its log. What it overrides:
  - what starts a run: the project's own checks, a save, a results folder per run;
  - what the thread calls: Project.run, that is run_topopt, with progress and should_stop;
  - how the design is drawn: elements by their nearest material's colour, not by density;
  - Stop: run_topopt is asked to stop before its next iteration and saves what it reached
    (PyTO-AD's Stop only cleared a flag the optimizer never read);
  - the end: the final design on the part, and the convergence, latent space and summary.

The project can still be edited while a run goes: the run has its own copy, taken at Start.
"""
import datetime
import os
import subprocess
import sys
import time

import numpy as np
from PyQt5 import QtCore, QtWidgets

from .panels import GREY, Info
from .pyto_ad import StructuralTopOptWindow

DRAW_EVERY = 1.5                                            # seconds between redraws of the design
RUN_HELP = ("Start optimizes the project as it is now: it is saved first, and the run keeps its own copy, "
            "so you can go on editing for the next run. Stop ends the run after the iteration under way "
            "and keeps the design reached so far. Each run has its own folder under results.")


def results_folder(project):
    """results/<project name>/<date and time>, next to the project file."""
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(project.base_dir, "results", project.name, stamp)


def nearest_material(z, z_real):
    """Index of the real material whose latent point is nearest each element's."""
    d = ((np.asarray(z)[:, None, :] - np.asarray(z_real)[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1)


class RunWindow(StructuralTopOptWindow):
    """PyTO-AD's Optimize window, running LSR MMTO's optimization (see the module docstring)."""

    def __init__(self, window):
        self.window, self.stop_requested, self.save_dir = window, False, None
        self.started, self.last_draw, self.history = 0.0, 0.0, {"objective": [], "constraints": []}
        super().__init__(window)
        self.setWindowTitle("Run")

    # ------------------------------------------------------------------ the strip
    def setup_ui(self):
        """In place of PyTO-AD's: no formulation of its own (the project's Formulation panel is the one)."""
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        layout = QtWidgets.QVBoxLayout(self)
        buttons = QtWidgets.QHBoxLayout()
        self.optimize_button = QtWidgets.QPushButton("Start")
        self.optimize_button.clicked.connect(self.start_optimization)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_optimization)
        self.stop_button.setEnabled(False)
        for widget in (self.optimize_button, self.stop_button):
            buttons.addWidget(widget)
        buttons.addStretch()
        buttons.addWidget(Info(RUN_HELP))
        layout.addLayout(buttons)
        self.status = QtWidgets.QLabel("Not started.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.tabs = QtWidgets.QTabWidget()
        self.history_figure = Figure(figsize=(4, 2.6), tight_layout=True)
        self.history_canvas = FigureCanvasQTAgg(self.history_figure)
        self.history_canvas.setMinimumHeight(200)
        self.tabs.addTab(self.history_canvas, "Progress")
        self.formulation = QtWidgets.QWidget()                     # PyTO-AD sends log lines to formulation.log_view
        self.formulation.log_view = QtWidgets.QPlainTextEdit()
        self.formulation.log_view.setReadOnly(True)
        self.tabs.addTab(self.formulation.log_view, "Log")
        layout.addWidget(self.tabs, stretch=1)

        self.show_each_iteration = True
        self.live_topology_check = QtWidgets.QCheckBox("Show the design while it runs")
        self.live_topology_check.setChecked(True)
        self.live_topology_check.toggled.connect(lambda on: setattr(self, "show_each_iteration", bool(on)))
        layout.addWidget(self.live_topology_check)

        after = QtWidgets.QHBoxLayout()                            # once a run has finished
        self.view_button = QtWidgets.QPushButton("Show setup")
        self.view_button.setCheckable(True)
        self.view_button.toggled.connect(self.show_setup)
        self.folder_button = QtWidgets.QPushButton("Open results folder")
        self.folder_button.clicked.connect(self.open_folder)
        for widget in (self.view_button, self.folder_button):
            widget.setVisible(False)
            after.addWidget(widget)
        after.addStretch()
        layout.addLayout(after)

    def problem_physics(self):
        return "structural"

    def selection_names(self):
        return list(self.window.project.selections)

    # ------------------------------------------------------------------ start, run, stop
    def start_optimization(self):
        """The project's checks, a save, then run_topopt in PyTO-AD's worker thread."""
        w = self.window
        errors = w.project.validate()
        if errors:
            self.status.setText("<span style='color:#b00000'>Cannot start:<br>" + "<br>".join(errors) + "</span>")
            return False
        if w.is_dirty() or w.path is None:
            answer = QtWidgets.QMessageBox.question(
                self, "Run", "The project is saved before a run, so the run matches the file. Save now?",
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Cancel)
            if answer != QtWidgets.QMessageBox.Save or not w.save_project():
                return False
        import threading
        self.project = w.project.copy()                            # the run's own copy: edits go to the next run
        self.save_dir = results_folder(self.project)
        self.stop_requested, self.history = False, {"objective": [], "constraints": []}
        self.started, self.last_draw = time.time(), 0.0
        self.max_iterations = int(self.project.settings["common"]["maxIterations"])
        self.colours = self._material_colours()
        self.optimization_running = True
        self.optimize_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        for widget in (self.view_button, self.folder_button):
            widget.setVisible(False)
        self.formulation.log_view.clear()
        self.status.setText("Starting: building the mesh and the material network...")
        w.message_text.append(f"Run started: results in {self.save_dir}")
        self.optimization_thread = threading.Thread(target=self.run_optimization, args=(None, None, None),
                                                    daemon=True)
        self.optimization_thread.start()
        return True

    def _run_optimization(self, plan, fe_solver, spec):
        """In the worker thread (PyTO-AD's run_optimization sends what is printed to the log)."""
        from project import run  # type: ignore
        success, error = False, ""
        try:
            run(self.project, save_dir=self.save_dir, progress=self._progress,
                should_stop=lambda: self.stop_requested)
            success = True
        except Exception as e:                                      # noqa: BLE001
            import traceback
            traceback.print_exc()
            error = str(e)
        self.optimization_done.emit(success, error, self.history, self.save_dir, None)

    def _progress(self, state):
        """After each iteration (worker thread): the plot always, the design at most every DRAW_EVERY s."""
        self.history["objective"].append(state["objective"])
        self.history["constraints"].append(state["constraints"])
        self.history["iteration"] = state["iteration"]
        self.history_update.emit({k: list(v) if isinstance(v, list) else v for k, v in self.history.items()})
        now = time.time()
        if self.show_each_iteration and now - self.last_draw >= DRAW_EVERY:
            self.last_draw = now
            self.optimization_update.emit(
                {"x": state["x_phys"], "material": nearest_material(state["z"], state["z_real"])},
                state["iteration"])

    def stop_optimization(self):
        """Ask run_topopt to stop before its next iteration; it saves the design reached so far."""
        if self.optimization_running:
            self.stop_requested = True
            self.stop_button.setEnabled(False)
            self.status.setText("Stopping after the iteration under way; the design so far is kept...")

    # ------------------------------------------------------------------ while it runs
    @QtCore.pyqtSlot(object)
    def update_history_plot(self, history):
        """The objective, and each constraint in its <= 0 form (met at or below the line)."""
        self.history_figure.clear()
        ax = self.history_figure.add_subplot(111)
        ax.plot(history["objective"], color="tab:blue")
        ax.set_xlabel("iteration")
        ax.set_ylabel("objective", color="tab:blue")
        cons = np.array(history["constraints"], dtype=float)
        if cons.size:
            ax2 = ax.twinx()
            for k in range(cons.shape[1]):
                ax2.plot(cons[:, k], linestyle=":", label=f"constraint {k + 1}")
            ax2.axhline(0.0, color="grey", linewidth=0.8)
            ax2.set_ylabel("constraint (met at or below 0)")
            ax2.legend(loc="center right", fontsize=8)
        self.history_canvas.draw_idle()
        i = history.get("iteration", 0)                            # 0 is the starting design
        elapsed = time.time() - self.started
        left = elapsed / (i + 1) * max(self.max_iterations - i, 0)
        if not self.stop_requested:
            self.status.setText(f"Iteration {i} / {self.max_iterations} · {_clock(elapsed)} so far "
                                f"· about {_clock(left)} left")

    @QtCore.pyqtSlot(object, int)
    def update_visualization(self, design, iteration):
        """The design on the part (main thread): solid elements as blocks in their material's colour."""
        self.draw_design(np.asarray(design["x"]) > 0.5, design["material"])

    def draw_design(self, solid, material):
        import pyvista as pv
        w = self.window
        mesh = w.preview_mesh()
        w.clear_preview()
        part = w.part_actor()
        if part is not None:
            part.SetVisibility(False)                              # the design in place of the part
        centres = np.asarray(mesh.elem_centers, dtype=float)[solid]
        if len(centres):
            cloud = pv.PolyData(centres)
            rgb = np.array([_rgb(self.colours[m % len(self.colours)]) for m in np.asarray(material)[solid]],
                           dtype=np.uint8)
            cloud.point_data["rgb"] = rgb
            h = np.asarray(mesh.elem_size, dtype=float)
            blocks = cloud.glyph(geom=pv.Cube(x_length=h[0], y_length=h[1], z_length=h[2]), scale=False,
                                 orient=False)
            w.plotter.add_mesh(blocks, scalars="rgb", rgb=True, pickable=False, name="mmto_design",
                               reset_camera=False)
        w.plotter.render()

    def _material_colours(self):
        from databases import material_display  # type: ignore
        try:
            colours, _ = material_display(self.project.database_paths()[0])
            return [c or "#808080" for c in colours]
        except Exception:                                           # noqa: BLE001
            return ["#808080"]

    # ------------------------------------------------------------------ the end
    @QtCore.pyqtSlot(bool, str, object, object, object)
    def optimization_completed(self, success, error, history, save_dir, _):
        self.optimization_running = False
        self.optimize_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        w = self.window
        if not success:
            self.status.setText(f"<span style='color:#b00000'>The run failed: {error}</span><br>"
                                f"The Log tab has the details.")
            w.message_text.append(f"Run failed: {error}")
            return
        how = "stopped" if self.stop_requested else "finished"
        self.status.setText(f"Run {how} after {max(len(history['objective']) - 1, 0)} iterations, "
                            f"{_clock(time.time() - self.started)}.")
        w.message_text.append(f"Run {how}: results in {save_dir}")
        self.view_button.setVisible(True)
        self.folder_button.setVisible(True)
        self.show_results(save_dir)

    def show_results(self, save_dir):
        """The final design on the part, and tabs for the convergence, the latent space, the summary."""
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from utils import figures, records, tables  # type: ignore
        try:
            run = records.load(os.path.join(save_dir, "run_data"))
        except Exception as e:                                      # noqa: BLE001
            self.status.setText(self.status.text() + f"<br>The results could not be read: {e}")
            return
        self.draw_design(np.asarray(run.solid, dtype=bool), np.asarray(run.material_indices))
        while self.tabs.count() > 2:                               # Progress and Log stay
            self.tabs.removeTab(2)
        for title, make in (("Convergence", lambda: figures.convergence(run)),
                            ("Latent space", lambda: figures.latent_scatter(run))):
            try:
                fig = make()
                self.tabs.addTab(FigureCanvasQTAgg(fig), title)
            except Exception as e:                                  # noqa: BLE001
                self.tabs.addTab(QtWidgets.QLabel(f"Could not draw: {e}"), title)
        summary = QtWidgets.QPlainTextEdit(str(tables.summary(run, show=False)))
        summary.setReadOnly(True)
        self.tabs.addTab(summary, "Summary")

    def show_setup(self, on):
        """Back to the part with its supports and loads, or the design again."""
        w = self.window
        self.view_button.setText("Show design" if on else "Show setup")
        part = w.part_actor()
        if part is not None:
            part.SetVisibility(on)
        if "mmto_design" in w.plotter.actors:
            w.plotter.actors["mmto_design"].SetVisibility(not on)
        w.plotter.render()

    def open_folder(self):
        if self.save_dir and os.path.isdir(self.save_dir):
            if sys.platform.startswith("win"):
                os.startfile(os.path.abspath(self.save_dir))        # noqa: S606
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", self.save_dir])

    def closeEvent(self, event):
        if self.optimization_running:
            answer = QtWidgets.QMessageBox.question(
                self, "Run", "A run is going. Stop it (the design so far is kept)?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            self.stop_optimization()
        QtWidgets.QDialog.closeEvent(self, event)


def _clock(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}" if seconds >= 3600 else \
        f"{seconds // 60}:{seconds % 60:02d}"


def _rgb(colour):
    colour = colour.lstrip("#")
    return [int(colour[i:i + 2], 16) for i in (0, 2, 4)]
