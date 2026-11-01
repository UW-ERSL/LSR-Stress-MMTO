"""
main_window.py
==============
The LSR MMTO window. It is PyTO-AD's MainWindow (viewer, sidebar, message panel) holding a
Project: File > New, Open, Save and Save As work on .mmto files through Project.load and
Project.save, so a project opened and saved with no edits is written back unchanged.

    python run_gui.py [project.mmto]        (from the repository folder)

The sidebar keeps PyTO-AD's Units, Geometry and Display Options. The panels for materials,
the formulation, the settings and the supports and loads come in the next steps.
"""
import json
import os
import sys

from PyQt5 import QtWidgets
from PyQt5.QtGui import QKeySequence

from .pyto_ad import (FILE_DIALOG_OPTIONS, UNIT_SYSTEMS, GeometryWindow, MainWindow, ProjectsWindow,
                      UnitsWindow)

APP = "LSR MMTO"
PROJECT_FILTER = "LSR MMTO project (*.mmto)"
SIDEBAR = ("Units", "Geometry", "Display Options")


def mmto_file_error(path):
    """Why the file at path cannot be opened in the GUI, or None. Only .mmto projects open here."""
    if not path.lower().endswith(".mmto"):
        return (f"{os.path.basename(path)} is not an LSR MMTO project (.mmto). A PyTO-AD .pyto file opens "
                f"in PyTO-AD, or in code with Project.load.")
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError) as e:
        return f"{os.path.basename(path)} could not be read: {e}"
    if not isinstance(data, dict) or "mmto" not in data:
        return f"{os.path.basename(path)} has no mmto section, so it is not an LSR MMTO project."
    return None


def triangle_selections(project):
    """Names of the selections made of STL triangle numbers (stl_faces, also inside a union, ...).
    On another STL the same numbers are other triangles."""
    def has(d):
        if isinstance(d, dict):
            return d.get("type") == "stl_faces" or any(has(v) for v in d.values())
        if isinstance(d, list):
            return any(has(v) for v in d)
        return False
    return [name for name, s in project.selections.items() if has(s.to_dict())]


def summary(project):
    """The project as lines of text for the message panel."""
    p = project
    lines = [f"Units: {p.units}" + ("  (none declared: Units sets one)" if p.units == "none" else ""),
             f"Database: {p.database or '(none)'}",
             f"Selections ({len(p.selections)}): " + (", ".join(f"{k} = {s}" for k, s in p.selections.items())
                                                       or "(none)")]
    bcs = [f"{bc['kind']} on {bc['on']} (" + ", ".join(f"{k}={v}" for k, v in bc.items() if k not in ("kind", "on"))
           + ")" for bc in p.bcs]
    lines.append(f"Supports and loads ({len(bcs)}): " + ("; ".join(bcs) or "(none)"))
    if p.keep_solid_names:
        lines.append("Keep solid: " + ", ".join(p.keep_solid_names))
    f = p.formulation
    obj = f.get("objective")
    lines.append("Objective: " + (f"{obj['sense']} {obj['expression']}" if obj else "(none)"))
    for c in f.get("constraints", []):
        lines.append(f"Constraint: {c['expression']} {c['op']} {c['bound']:g}")
    lines.append(f"Mesh: {p.n_dof} dofs" + (", single layer" if p.single_layer else ""))
    return lines


class MMTOWindow(MainWindow):
    """PyTO-AD's MainWindow with a Project, a File menu, and the sidebar cut to SIDEBAR."""

    BUTTON_CONFIG = [c for c in MainWindow.BUTTON_CONFIG if c["name"] in SIDEBAR]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project, self.path, self._saved = None, None, None
        self.message_text.setText(f"{APP}. File > New starts a project from an STL; File > Open opens a "
                                  f".mmto project.")
        self._build_menu()
        self._refresh_title()

    # ------------------------------------------------------------------ the window
    def create_status_bar(self):
        self.statusBar().addWidget(QtWidgets.QLabel(f"{APP}, built on PyTO-AD."))

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")
        for text, key, slot in (("&New...", QKeySequence.New, self.new_project),
                                ("&Open...", QKeySequence.Open, self.open_project),
                                ("&Save", QKeySequence.Save, self.save_project),
                                ("Save &As...", QKeySequence.SaveAs, self.save_project_as)):
            action = file_menu.addAction(text)
            action.setShortcut(key)
            action.triggered.connect(lambda _=False, s=slot: s())
        file_menu.addSeparator()
        file_menu.addAction("E&xit").triggered.connect(self.close)
        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(f"About {APP}").triggered.connect(self._about)

    def _about(self):
        QtWidgets.QMessageBox.information(
            self, f"About {APP}", f"{APP}: multi-material topology optimization with a latent material "
                                  f"space, built on PyTO-AD.")

    # ------------------------------------------------------------------ unsaved changes
    def _state(self):
        p = self.project
        return json.dumps([p.to_dict(), p.units, os.path.abspath(p.stl_path())], sort_keys=True, default=str)

    def is_dirty(self):
        """True when the project has changes not in its file (always, for a project never saved)."""
        return self.project is not None and (self.path is None or self._state() != self._saved)

    def _refresh_title(self):
        if self.project is None:
            self.setWindowTitle(APP)
            return
        name = os.path.splitext(os.path.basename(self.path))[0] if self.path else self.project.name
        self.setWindowTitle(f"{name}{'*' if self.is_dirty() else ''} - {APP}")

    def _ask_to_save(self):
        """False when the user cancels. Asks only when there are unsaved changes."""
        if not self.is_dirty():
            return True
        answer = QtWidgets.QMessageBox.question(
            self, APP, f"Save the changes to {self.windowTitle().split('*')[0]}?",
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Save)
        if answer == QtWidgets.QMessageBox.Save:
            return self.save_project()
        return answer == QtWidgets.QMessageBox.Discard

    def closeEvent(self, event):
        if self._ask_to_save():
            event.accept()
        else:
            event.ignore()

    # ------------------------------------------------------------------ New, Open, Save
    def _folder(self):
        return os.path.dirname(self.path) if self.path else os.getcwd()

    def new_project(self, stl=None):
        """A project on an STL, every setting at its default; not saved until Save."""
        from project import Project  # type: ignore
        if not self._ask_to_save():
            return False
        if stl is None:
            stl, _ = QtWidgets.QFileDialog.getOpenFileName(self, "New project: choose the STL", self._folder(),
                                                           "STL Files (*.stl *.STL)", options=FILE_DIALOG_OPTIONS)
            if not stl:
                return False
        stl = os.path.abspath(stl)
        p = Project(os.path.splitext(os.path.basename(stl))[0], stl=stl, base_dir=os.path.dirname(stl))
        p.complete_settings()
        self._show(p, None)
        return True

    def open_project(self, path=None):
        from project import Project  # type: ignore
        if not self._ask_to_save():
            return False
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open project", self._folder(), PROJECT_FILTER,
                                                            options=FILE_DIALOG_OPTIONS)
            if not path:
                return False
        error = mmto_file_error(path)
        if error is None:
            try:
                p = Project.load(path)
            except Exception as e:                                  # noqa: BLE001
                error = f"{os.path.basename(path)} could not be loaded: {e}"
        if error:
            QtWidgets.QMessageBox.warning(self, "Open project", error)
            self.message_text.append(error)
            return False
        self._show(p, os.path.abspath(path))
        return True

    def save_project(self):
        if self.project is None:
            return False
        if self.path is None:
            return self.save_project_as()
        return self._write(self.path)

    def save_project_as(self, path=None):
        if self.project is None:
            return False
        if path is None:
            default = self.path or os.path.join(self._folder(), f"{self.project.name}.mmto")
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save project", default, PROJECT_FILTER,
                                                            options=FILE_DIALOG_OPTIONS)
            if not path:
                return False
        if not path.lower().endswith(".mmto"):
            path += ".mmto"
        return self._write(path)

    def _write(self, path):
        try:
            self.project.save(path)
        except Exception as e:                                      # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Save project", f"Not saved: {e}")
            return False
        self.path, self._saved = os.path.abspath(path), self._state()
        self.message_text.append(f"Saved {self.path}")
        self._refresh_title()
        return True

    def _show(self, project, path):
        """Make project the one in the window: PyTO-AD's state reset, units synced, the STL drawn."""
        self.project, self.path = project, path
        self._saved = self._state() if path else None
        filled = project.complete_settings()                        # a file missing settings: filled, unsaved
        self.init_state()                                           # PyTO-AD's own fresh state
        self._units_to_window()
        self.message_text.clear()
        self.message_text.append(f"Project {project.name}" + (f": {path}" if path else " (not saved yet)"))
        if os.path.isfile(project.stl_path()):
            loader = ProjectsWindow(self)                           # its load_geometry draws the STL, picking on
            loader.load_geometry(project.stl_path())
            loader.deleteLater()
        else:
            self.plotter.clear_actors()
            self.message_text.append(f"STL not found: {project.stl_path()}")
        for line in summary(project):
            self.message_text.append(line)
        if filled:
            self.message_text.append(f"{len(filled)} settings missing from the file were set to their defaults "
                                     f"({', '.join(filled)}); Save writes them.")
        for e in project.validate():
            self.message_text.append(f"Error: {e}")
        for w in project.warnings():
            self.message_text.append(f"Warning: {w}")
        self._refresh_title()

    # ------------------------------------------------------------------ units
    def _units_to_window(self):
        """PyTO-AD's Settings take the project's unit system (MKS for a project that declares none,
        since the Units dialog has no 'none')."""
        s = self.settings
        units = self.project.units if self.project.units in UNIT_SYSTEMS else "MKS"
        s.update_settings(units, s.temperature_unit, s.angle_unit)

    def open_units_window(self):
        """PyTO-AD's Units dialog (as MainWindow.open_units_window); Apply sets the project's units.
        Nothing is converted: the numbers stay as written, the unit warnings are checked again."""
        dialog = UnitsWindow(self, self.settings)
        if dialog.exec_() and self.project is not None and self.project.units != self.settings.unit_system:
            self.project.units = self.settings.unit_system
            self.message_text.append(f"Project units: {self.project.units}. Numbers are not converted.")
            for w in self.project.warnings():
                self.message_text.append(f"Warning: {w}")
            self._refresh_title()

    # ------------------------------------------------------------------ geometry
    def open_geometry_window(self):
        """With no project, a new one. Otherwise PyTO-AD's Geometry dialog changes the project's STL,
        after a warning when selections are triangle numbers of the current one."""
        if self.project is None:
            self.new_project()
            return
        names = triangle_selections(self.project)
        if names:
            answer = QtWidgets.QMessageBox.question(
                self, "Change the STL",
                f"These selections are triangle numbers of the current STL: {', '.join(names)}. On another "
                f"STL the same numbers are other triangles. Change the STL anyway?")
            if answer != QtWidgets.QMessageBox.Yes:
                return
        before = self.stl_geom.file_path if self.stl_geom else None
        dialog = GeometryWindow(self)
        dialog.exec_()
        after = self.stl_geom.file_path if self.stl_geom else None
        if after and after != before:
            self.project.stl = os.path.abspath(after)
            if names:
                self.message_text.append(f"Check the selections {', '.join(names)}: they are triangle numbers "
                                         f"of the old STL.")
            self._refresh_title()


def main(argv=()):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyleSheet("* { font-size: 10pt; }")                    # as PyTO-AD's main()
    window = MMTOWindow()
    window.show()
    if argv:
        window.open_project(argv[0])
    return app.exec_()
