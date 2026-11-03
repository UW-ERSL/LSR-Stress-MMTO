"""
run_panel.py
============
The Run strip: an optimization of the project, started from the window.

RunWindow is PyTO-AD's Optimize window (StructuralTopOptWindow): its worker thread, its Qt
signals, its live plot, its live redraw of the design and its log. What it overrides:
  - what starts a run: the project's own checks, a save, a results folder per run;
  - what the thread calls: Project.run, that is _run_topopt, with progress and should_stop;
  - how the design is drawn: each element in its nearest material's colour, at a strength that
    follows its density (design.py), as the window's design (Setup and Design views);
  - Stop: _run_topopt is asked to stop before its next iteration and saves what it reached
    (PyTO-AD's Stop only cleared a flag the optimizer never read);
  - the end: the final design on the part, the convergence, latent space and summary, and the
    run's folder: open it, show it in Results, or delete it.
While the material network trains, before the optimization starts, the status line counts
its epochs from the lines the training prints.

The project can still be edited while a run goes: the run has its own copy, taken at Start.
"""
import os
import re
import subprocess
import sys
import time

import numpy as np
from PyQt5 import QtCore, QtWidgets

from . import draw_design
from .panel_widgets import Info
from .pyto_ad_imports import StructuralTopOptWindow

DRAW_EVERY = 1.5                                            # seconds between redraws of the design
LATENT_POINTS = 5000                                        # the most elements the live latent space draws
RUN_HELP = ("Start optimizes the project as it is now: it is saved first, and the run keeps its own copy, "
            "so you can go on editing for the next run. Stop ends the run after the iteration under way "
            "and keeps the design reached so far. Each run has its own folder in the project's results "
            "folder.\n\nWhile it runs, each element is drawn in its material's colour, pale while it is "
            "half solid and full colour once solid; elements that are almost empty are not drawn.\n\n"
            "The Latent space tab shows where the solid elements' materials are heading: red points for "
            "the elements, black stars for the real materials. With one latent dimension it counts the "
            "elements along the line; with more than three it shows a flat view, and its title says for "
            "how many of the elements that view shows the right nearest material.")
EPOCH = re.compile(r"^(?:\[(?:STD|SAM)\] )?Epoch (\d+):")   # what the network's training prints


def results_folder(project):
    """The folder for the next run's results (project_folder.new_run_folder)."""
    from lsr_mmto.problem.project_folder import new_run_folder
    return new_run_folder(project)


def open_folder(path):
    """A folder opened in the file manager."""
    if path and os.path.isdir(path):
        if sys.platform.startswith("win"):
            os.startfile(os.path.abspath(path))                     # noqa: S606
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])


def ask_delete_run(parent, folder):
    """A run's folder deleted after a yes; True when it was."""
    from lsr_mmto.problem.project_folder import delete_run, run_facts
    when = run_facts(folder)[0]
    answer = QtWidgets.QMessageBox.question(
        parent, "Delete this run", f"Delete the run of {when}? Its folder, with everything in it, is removed. "
                                   f"This cannot be undone.",
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
    if answer != QtWidgets.QMessageBox.Yes:
        return False
    try:
        delete_run(folder)
    except (OSError, ValueError) as e:
        QtWidgets.QMessageBox.warning(parent, "Delete this run", f"Not deleted: {e}")
        return False
    return True


def result_tabs(tabs, run, keep=0):
    """Convergence, Latent space and Summary tabs of a saved run, after the first keep tabs."""
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from lsr_mmto.results import run_plots, tables
    while tabs.count() > keep:
        tabs.removeTab(keep)
    for title, make in (("Convergence", lambda: run_plots.convergence(run)),
                        ("Latent space", lambda: run_plots.latent_scatter(run))):
        try:
            tabs.addTab(FigureCanvasQTAgg(make()), title)
        except Exception as e:                                      # noqa: BLE001
            tabs.addTab(QtWidgets.QLabel(f"Could not draw: {e}"), title)
    tabs.addTab(table_view(str(tables.summary(run, show=False))), "Summary")


def table_view(text):
    """A table as text, in a fixed-width font and unwrapped, so its columns line up (scroll to the side)."""
    from PyQt5 import QtGui
    view = QtWidgets.QPlainTextEdit(text)
    view.setReadOnly(True)
    view.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
    font = QtGui.QFont("Consolas")                                 # Windows'; elsewhere the hint finds one
    font.setStyleHint(QtGui.QFont.TypeWriter)
    view.setFont(font)
    return view


def latent_frame(state, order):
    """What the live latent space draws after an iteration (worker thread): the latent points of the
    solid elements (density 0.5 or more), at most LATENT_POINTS of them taken in the run's fixed random
    order, so that the same elements stay drawn from one frame to the next."""
    solid = np.asarray(state["x_phys"]).ravel()[order] >= 0.5
    pick = order[solid][:LATENT_POINTS]
    z = np.asarray(state["z"], dtype=float)
    return {"z": z.reshape(len(z), -1)[pick], "z_real": np.asarray(state["z_real"], dtype=float),
            "iteration": int(state["iteration"])}


def nearest_material(z, z_real):
    """Index of the real material whose latent point is nearest each element's."""
    d = ((np.asarray(z)[:, None, :] - np.asarray(z_real)[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1)


class RunWindow(StructuralTopOptWindow):
    """PyTO-AD's Optimize window, running LSR MMTO's optimization (see the module docstring)."""
    latent_update = QtCore.pyqtSignal(object)                  # a frame of the live latent space
    network_question = QtCore.pyqtSignal(str)                  # the network warning, asked in the window

    def __init__(self, window):
        self.window, self.stop_requested, self.save_dir = window, False, None
        self.latent_shown, self.last_latent, self.latent_dragging, self.latent_pending = False, 0.0, False, None
        self.latent_order = None
        self.started, self.last_draw, self.history = 0.0, 0.0, {"objective": [], "constraints": []}
        self.first_frame, self.optimizing = True, False
        super().__init__(window)
        self.setWindowTitle("Run")
        self.log_line.connect(self._training_progress, QtCore.Qt.QueuedConnection)
        self.latent_update.connect(self.update_latent_plot, QtCore.Qt.QueuedConnection)
        self.network_question.connect(self._ask_network, QtCore.Qt.BlockingQueuedConnection)

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
        self.latent_figure = Figure(figsize=(4, 2.6))              # the live latent space: a tab while a run goes
        self.latent_canvas = FigureCanvasQTAgg(self.latent_figure)
        self.latent_canvas.setMinimumHeight(200)
        self.latent_canvas.mpl_connect("button_press_event", lambda _: setattr(self, "latent_dragging", True))
        self.latent_canvas.mpl_connect("button_release_event", lambda _: self._latent_released())
        self.tabs.currentChanged.connect(
            lambda _: setattr(self, "latent_shown", self.tabs.currentWidget() is self.latent_canvas))
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
        self.view_button = self.window.view_button()               # Show setup / Show design
        after.addWidget(self.view_button)
        self.folder_button = QtWidgets.QPushButton("Open results folder")
        self.folder_button.clicked.connect(self.open_folder)
        self.results_button = QtWidgets.QPushButton("Open in Results")
        self.results_button.clicked.connect(lambda: self.window.open_results_panel(self.save_dir))
        self.delete_button = QtWidgets.QPushButton("Delete this run")
        self.delete_button.clicked.connect(self.delete_run)
        self.run_buttons = (self.folder_button, self.results_button, self.delete_button)
        for widget in self.run_buttons:
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
        """The project's checks, a save, then _run_topopt in PyTO-AD's worker thread."""
        w = self.window
        errors = w.project.validate()
        if errors:
            self.status.setText("<span style='color:#b00000'>Cannot start:<br>" + "<br>".join(errors) + "</span>")
            return False
        if w.template:
            answer = QtWidgets.QMessageBox.question(
                self, "Run", "This is an example: it is saved as a project of your own before it runs, so the "
                             "example stays as it is and the results go in your project's folder. Save now?",
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Cancel)
            if answer != QtWidgets.QMessageBox.Save or not w.save_project():
                return False
        elif w.is_dirty() or w.path is None:
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
        self.first_frame, self.optimizing = True, False
        self.max_iterations = int(self.project.settings["common"]["maxIterations"])
        self.epochs = int(self.project.settings["vae"].get("numEpochs") or 0)
        self.colours = self._material_colours()
        self.optimization_running = True
        self.optimize_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        for widget in self.run_buttons:
            widget.setVisible(False)
        self.formulation.log_view.clear()
        self._start_live_latent()
        self.status.setText("Starting: building the mesh and the material network...")
        w.message_text.append(f"Run started: results in {self.save_dir}")
        self.optimization_thread = threading.Thread(target=self.run_optimization, args=(None, None, None),
                                                    daemon=True)
        self.optimization_thread.start()
        return True

    def _run_optimization(self, plan, fe_solver, spec):
        """In the worker thread (PyTO-AD's run_optimization sends what is printed to the log)."""
        from lsr_mmto.materials.encoder import NetworkRunCancelled, TrainingStopped
        from lsr_mmto.problem.project import run
        success, error = False, ""
        self.stopped_training, self.network_cancelled = False, False
        try:
            run(self.project, save_dir=self.save_dir, progress=self._progress,
                should_stop=lambda: self.stop_requested, on_network_error=self._network_error)
            success = True
        except TrainingStopped:                                     # Stop before the first iteration
            self.stopped_training = True
        except NetworkRunCancelled:                                 # Cancel on the network warning: no run
            self.network_cancelled = True
        except Exception as e:                                      # noqa: BLE001
            import traceback
            traceback.print_exc()
            error = str(e)
        self.optimization_done.emit(success, error, self.history, self.save_dir, None)

    def _network_error(self, warning):
        """In the worker thread: the network warning asked in the window (_ask_network); True to go on."""
        self.network_question.emit(warning)                        # returns once it is answered
        return self.network_answer

    @QtCore.pyqtSlot(str)
    def _ask_network(self, warning):
        """Continue or Cancel, with the warning; the window's thread."""
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, "Material network",
                                    warning[0].upper() + warning[1:] + ".\n\nThe run would use material "
                                    "properties this far from the database. Run anyway?", parent=self)
        box.setDetailedText("To train the network again, untick Use the trained network in Settings. "
                            "The warning limit is Warn above network error [%] in Settings.")
        go = box.addButton("Continue", QtWidgets.QMessageBox.AcceptRole)
        box.addButton(QtWidgets.QMessageBox.Cancel)
        box.exec_()
        self.network_answer = box.clickedButton() is go

    def _progress(self, state):
        """After each iteration (worker thread): the plot always, the design at most every DRAW_EVERY s."""
        self.optimizing = True                                     # training, if any, is over
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
        if self.latent_shown and now - self.last_latent >= DRAW_EVERY:
            self.last_latent = now
            if self.latent_order is None:
                self.latent_order = np.random.default_rng(0).permutation(np.asarray(state["x_phys"]).size)
            self.latent_update.emit(latent_frame(state, self.latent_order))

    @QtCore.pyqtSlot(str)
    def _training_progress(self, line):
        """While the material network trains (before the first iteration): its epoch in the status line."""
        m = EPOCH.match(line)
        if m is None or self.optimizing or not self.optimization_running or self.stop_requested:
            return
        epoch = int(m.group(1))
        total = f" / {self.epochs:,}" if self.epochs else ""
        self.status.setText(f"Training the material network: epoch {epoch:,}{total}")

    def stop_optimization(self):
        """Ask _run_topopt to stop before its next iteration; it saves the design reached so far."""
        if self.optimization_running:
            self.stop_requested = True
            self.stop_button.setEnabled(False)
            self.status.setText("Stopping after the iteration under way; the design so far is kept..." if self.optimizing
                                else "Stopping...")

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
    def update_visualization(self, frame, iteration):
        """The design so far (main thread), as the window's design: shown at the first frame of a run,
        later only while the Design view is on (Show setup keeps the part in view)."""
        mesh = draw_design.mesh_of(self.window, self.project)           # the run's own copy: edits wait
        x, material, colours = np.asarray(frame["x"]), np.asarray(frame["material"]), self.colours
        self.window.set_design(lambda w: draw_design.live(w, mesh, x, material, colours), show=self.first_frame)
        self.first_frame = False

    # ------------------------------------------------------------------ the live latent space
    def _start_live_latent(self):
        """A run starts: the last run's result tabs go, and the live latent space comes in beside Progress."""
        from lsr_mmto.materials.databases import material_display
        while self.tabs.count() > 2:
            self.tabs.removeTab(2)
        self.latent_order, self.latent_pca, self.latent_limits, self.latent_pending = None, None, None, None
        try:
            self.latent_names = list(material_display(self.project.database_paths()[0])[1])
        except Exception:                                           # noqa: BLE001
            self.latent_names = []
        self.latent_figure.clear()
        self.latent_figure.text(0.5, 0.5, "The latent space appears with the first iteration.", ha="center",
                                va="center", color="#555555")
        self.latent_canvas.draw_idle()
        if self.tabs.indexOf(self.latent_canvas) < 0:
            self.tabs.insertTab(1, self.latent_canvas, "Latent space")

    def _end_live_latent(self):
        """The run is over: the live tab goes (the run's own Latent space tab follows)."""
        i = self.tabs.indexOf(self.latent_canvas)
        if i >= 0:
            self.tabs.removeTab(i)
        self.latent_shown = False

    def _latent_released(self):
        self.latent_dragging = False
        if self.latent_pending is not None:
            frame, self.latent_pending = self.latent_pending, None
            self.update_latent_plot(frame)

    @QtCore.pyqtSlot(object)
    def update_latent_plot(self, frame):
        """A frame of the live latent space (main thread), drawn as the run's Latent space tab draws it.
        Above three latent dimensions the projection is fitted once, on the real materials, and the axes
        only ever widen, so the picture does not jump; a 3D view keeps the angle it was turned to, and
        is not redrawn while the mouse is turning it."""
        from lsr_mmto.results import run_plots
        if self.tabs.indexOf(self.latent_canvas) < 0:
            return
        if self.latent_dragging:
            self.latent_pending = frame
            return
        z, zr = frame["z"], frame["z_real"]
        if self.latent_pca is None and zr.shape[1] > 3:
            self.latent_pca = run_plots.latent_projection(zr)
        pca = self.latent_pca
        drawn = np.vstack([pca.transform(z) if pca is not None and len(z) else z.reshape(-1, zr.shape[1])[:, :3],
                           pca.transform(zr) if pca is not None else zr])
        lo, hi = drawn.min(axis=0), drawn.max(axis=0)
        pad = 0.05 * np.maximum(hi - lo, 1e-9)
        lo, hi = lo - pad, hi + pad
        if self.latent_limits is not None:
            lo, hi = np.minimum(lo, self.latent_limits[0]), np.maximum(hi, self.latent_limits[1])
        self.latent_limits = (lo, hi)
        old = self.latent_figure.axes
        view = (old[0].elev, old[0].azim) if old and hasattr(old[0], "elev") else None
        self.latent_figure.clear()
        names = self.latent_names + [str(i) for i in range(len(self.latent_names), zr.shape[0])]
        ax = run_plots.latent_axes(self.latent_figure, z, zr, names[:zr.shape[0]], pca=pca, small=True,
                                 limits=list(zip(lo, hi)), view=view)
        agreement = run_plots.latent_agreement(pca, z, zr) if pca is not None else None
        ax.set_title(f"Iteration {frame['iteration']}: " + run_plots.latent_title(zr.shape[1], agreement), fontsize=9)
        self.latent_figure.tight_layout()
        self.latent_canvas.draw_idle()

    def _material_colours(self):
        from lsr_mmto.materials.databases import material_display
        try:
            colours, _ = material_display(self.project.database_paths()[0])
            return [c or "#808080" for c in colours]
        except Exception:                                           # noqa: BLE001
            return ["#808080"]

    # ------------------------------------------------------------------ the end
    @QtCore.pyqtSlot(bool, str, object, object, object)
    def optimization_completed(self, success, error, history, save_dir, _):
        self.optimization_running = False
        self._end_live_latent()
        self.optimize_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        w = self.window
        cancelled = getattr(self, "network_cancelled", False)
        if getattr(self, "stopped_training", False) or cancelled:   # nothing to keep: no design
            from lsr_mmto.problem.project_folder import is_run
            if self.save_dir and os.path.isdir(self.save_dir) and not is_run(self.save_dir):
                import shutil
                shutil.rmtree(self.save_dir, ignore_errors=True)
            if cancelled:
                self.status.setText("Not run: cancelled at the material network warning. No design was made.")
                w.message_text.append("Run cancelled at the material network warning; no design was made.")
                return
            self.status.setText("Stopped while training the material network. Nothing was saved.")
            w.message_text.append("Run stopped while training the material network; nothing was saved.")
            return
        if not success:
            self.status.setText(f"<span style='color:#b00000'>The run failed: {error}</span><br>"
                                f"The Log tab has the details.")
            w.message_text.append(f"Run failed: {error}")
            return
        how = "stopped" if self.stop_requested else "finished"
        self.status.setText(f"Run {how} after {max(len(history['objective']) - 1, 0)} iterations, "
                            f"{_clock(time.time() - self.started)}.")
        w.message_text.append(f"Run {how}: results in {save_dir}")
        for widget in self.run_buttons:
            widget.setVisible(True)
        self.show_results(save_dir)
        results = getattr(w, "results_window", None)
        if results is not None:
            try:
                results.refresh()                                  # the new run in its list
            except RuntimeError:
                pass

    def show_results(self, save_dir):
        """The final design on the part, and tabs for the convergence, the latent space, the summary."""
        from lsr_mmto.results import load_results
        try:
            run = load_results.load(os.path.join(save_dir, "run_data"))
        except Exception as e:                                      # noqa: BLE001
            self.status.setText(self.status.text() + f"<br>The results could not be read: {e}")
            return
        mesh = draw_design.mesh_of(self.window, self.project)
        if int(mesh.num_elems) == int(run.n_elems):
            self.window.set_design(lambda w: draw_design.record(w, run, mesh, "Material"))
        result_tabs(self.tabs, run, keep=2)                        # Progress and Log stay

    def open_folder(self):
        open_folder(self.save_dir)

    def delete_run(self):
        """The run just made deleted, after a yes; its design goes too."""
        if self.save_dir and ask_delete_run(self, self.save_dir):
            self.window.message_text.append(f"Deleted the run {self.save_dir}")
            for widget in self.run_buttons:
                widget.setVisible(False)
            while self.tabs.count() > 2:
                self.tabs.removeTab(2)
            self.status.setText("The run was deleted.")
            self.window.show_setup_view()
            self.window.set_design(None, show=False)
            results = getattr(self.window, "results_window", None)
            if results is not None:
                try:
                    results.refresh()
                except RuntimeError:
                    pass

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
