"""
main_window.py
==============
The LSR MMTO window. It is PyTO-AD's MainWindow (viewer, sidebar, message panel) holding a
Project: File > New, Open, Save and Save As work on .mmto files through Project.load and
Project.save, so a project opened and saved with no edits is written back unchanged.

    python run_gui.py [project.mmto]        (from the repository folder)

The sidebar keeps PyTO-AD's Units, Geometry and Display Options, and adds Materials,
Formulation and Settings (panels.py) and Supports and loads (supports.py). Picking with the
mouse is PyTO-AD's Structural Loads window; each Apply in it is adopted into the project.
"""
import json
import os
import re
import sys

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QKeySequence

from .panels import FormulationPanel, MaterialsPanel, SettingsPanel
from .supports import KEEP_COLOUR, LOAD_COLOUR, OTHER_COLOUR, SUPPORT_COLOUR, SupportsPanel, bc_name, bc_value
from .pyto_ad import (FILE_DIALOG_OPTIONS, UNIT_SYSTEMS, GeometryWindow, MainWindow, ProjectsWindow,
                      StructuralLoadsWindow, UnitsWindow)

APP = "LSR MMTO"
PROJECT_FILTER = "LSR MMTO project (*.mmto)"
SIDEBAR = ("Units", "Geometry", "Materials", "Supports and loads", "Formulation", "Settings", "Display Options")
PANELS = {"Materials": MaterialsPanel, "Formulation": FormulationPanel, "Settings": SettingsPanel}


class Strip(QtWidgets.QDockWidget):
    """A non-modal dialog docked on the right edge of the window, so it does not cover the 3D view.
    Its own Close, or the strip's x, closes both; dragging the strip's title bar floats it."""

    def __init__(self, dialog, window, attr):
        super().__init__(dialog.windowTitle(), window)
        self.is_open = True                                         # also while behind another strip's tab
        self.window_, self.attr = window, attr                      # window.<attr> is the dialog while open
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        dialog.setWindowFlags(QtCore.Qt.Widget)
        self.setWidget(dialog)
        dialog.finished.connect(self.close)
        window.addDockWidget(QtCore.Qt.RightDockWidgetArea, self)
        dialog.show()
        self.show()

    def closeEvent(self, event):
        dialog = self.widget()
        self.is_open = False
        if getattr(self.window_, self.attr, None) is dialog:       # closed is gone: nothing may call it after
            delattr(self.window_, self.attr)                        # (Units' Apply updates an open Pick window)
        if dialog is not None and dialog.isVisible():
            dialog.close()                                          # its own clean-up (Pick: the node dots)
        super().closeEvent(event)


def in_open_strip(widget):
    """True when widget is docked in a strip that is open, in front or behind a tab."""
    try:
        strip = widget.parentWidget() if widget is not None else None    # (the Pick window's .parent is
        return isinstance(strip, Strip) and strip.is_open                # the main window, not a method)
    except RuntimeError:                                            # already deleted
        return False


class PickWindow(StructuralLoadsWindow):
    """Picking supports and loads with the mouse: Selection, Type, the force, Apply. Each Apply goes
    straight into the project (MMTOWindow.adopt_picks), so there is no list of applied loads here, no
    load set (one load case) and no torque (not one of LSR MMTO's loads)."""

    def init_window(self):
        super().init_window()
        self.setWindowTitle("Pick")

    def create_load_set_control(self, parent_layout):
        self.load_set_spin = QtWidgets.QSpinBox(self)               # read when a load is applied: always 0
        self.load_set_spin.setVisible(False)

    def create_load_type_control(self, parent_layout):
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Type"))
        self.load_type = QtWidgets.QComboBox()
        self.load_type.addItems([k for k in self.LOAD_TYPES if k != "Torque"])
        layout.addWidget(self.load_type)
        parent_layout.addLayout(layout)

    def create_buttons(self, parent_layout):
        self.apply_button = QtWidgets.QPushButton("Apply")
        parent_layout.addWidget(self.apply_button)

    def init_ui(self):
        super().init_ui()
        self.applied_list.parent().setVisible(False)                # always empty: Apply adds to the project
        hint = QtWidgets.QLabel("Click faces on the part, choose the type and value, then Apply.")
        hint.setWordWrap(True)
        self.layout().insertWidget(0, hint)
        self.layout().addStretch()


def mmto_file_error(path):
    """Why the file at path cannot be opened in the GUI, or None. Only .mmto projects open here."""
    if not path.lower().endswith(".mmto"):
        return f"{os.path.basename(path)} is not an LSR MMTO project (.mmto)."
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


class MMTOWindow(MainWindow):
    """PyTO-AD's MainWindow with a Project, a File menu, and the sidebar of SIDEBAR: PyTO-AD's
    buttons where it has them, ours (PANELS) for the rest."""

    BUTTON_CONFIG = [next((c for c in MainWindow.BUTTON_CONFIG if c["name"] == name),
                          {"name": name, "icon": "arrow", "always_enabled": True,
                           "handler": f"open_{name.lower().replace(' ', '_')}_panel"}) for name in SIDEBAR]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project, self.path, self._saved = None, None, None
        self._mesh, self._mesh_key, self.preview_on = None, None, False
        self._said = set()                                          # errors and warnings already shown
        self.message_text.setText(f"{APP}. File > New starts a project from an STL; File > Open opens a "
                                  f".mmto project.")
        self._build_menu()
        self._refresh_title()

    # ------------------------------------------------------------------ the window
    def create_status_bar(self):
        """The name on the left; on the right, the point under the mouse and the Ruler tick."""
        self.statusBar().addWidget(QtWidgets.QLabel(APP))
        self.coords = QtWidgets.QLabel("")
        self.coords.setStyleSheet("color: #666666;")
        self.coords.setMinimumWidth(240)
        self.ruler_box = QtWidgets.QCheckBox("Ruler")
        self.ruler_box.setToolTip("Show scales along the part, in the project's length unit")
        self.ruler_box.toggled.connect(lambda on: self.draw_ruler())
        self.statusBar().addPermanentWidget(self.coords)
        self.statusBar().addPermanentWidget(self.ruler_box)
        self._rulers = []
        self._picker = None
        QtCore.QTimer.singleShot(0, self._track_mouse)              # once the 3D view exists

    # ------------------------------------------------------------------ scale: the ruler and the mouse
    def length_unit(self):
        from .supports import field_unit
        return field_unit("length", self.project.units) if self.project is not None else ""

    def part_bounds(self):
        """(lowest, highest) corner of the part's STL, or None with no part."""
        if not getattr(self, "stl_geom", None):
            return None
        p = np.asarray(self.stl_geom.mesh.vectors, dtype=float).reshape(-1, 3)
        return p.min(axis=0), p.max(axis=0)

    def part_extent_text(self):
        """The part's extent in words, for the region form: x 0 to 200, y 0 to 200, z 0 to 1 mm."""
        b = self.part_bounds()
        if b is None:
            return ""
        lo, hi = b
        parts = ", ".join(f"{a} {lo[k]:.4g} to {hi[k]:.4g}" for k, a in enumerate("xyz"))
        return f"The part spans {parts} {self.length_unit()}".rstrip() + "."

    def draw_ruler(self):
        """Scales along the part's lowest x and y edges (and z, when the part has thickness), on or off
        with the Ruler tick."""
        for actor in self._rulers:
            self.plotter.remove_actor(actor, reset_camera=False)
        self._rulers = []
        b = self.part_bounds()
        if b is not None and self.ruler_box.isChecked():
            lo, hi = b
            size = float(np.max(hi - lo))
            off = 0.03 * size
            style = dict(number_labels=5, font_size_factor=0.6, label_size_factor=1.3, tick_length=3,
                         label_color="#8a8a8a", tick_color="#b5b5b5",
                         label_format="%.0f" if size >= 20 else "%.3g")
            x0, y0, z1 = lo[0] - off, lo[1] - off, hi[2]
            lines = [([lo[0], y0, z1], [hi[0], y0, z1], False, self.length_unit()),
                     ([x0, hi[1], z1], [x0, lo[1], z1], True, "")]
            if hi[2] - lo[2] > 0.05 * size:                         # a plate shows no z scale
                lines.append(([x0, y0, lo[2]], [x0, y0, hi[2]], False, ""))
            for a, b_, flip, title in lines:
                actor = self.plotter.add_ruler(a, b_, flip_range=flip, title=title, **style)
                for prop in (actor.GetLabelTextProperty(), actor.GetTitleTextProperty()):
                    prop.ItalicOff()
                    prop.BoldOff()
                    prop.ShadowOff()
                actor.SetPickable(False)
                self._rulers.append(actor)
        self.plotter.render()

    def _track_mouse(self):
        """The point of the part under the mouse, in the status bar."""
        import vtk
        try:
            self._picker = vtk.vtkCellPicker()
            self._picker.SetTolerance(0.0005)
            self.plotter.iren.add_observer("MouseMoveEvent", lambda *_: self.show_point_under_mouse())
        except Exception:                                           # noqa: BLE001  (no 3D view: no readout)
            self._picker = None

    def show_point_under_mouse(self, position=None):
        if self._picker is None or self.part_bounds() is None:
            self.coords.setText("")
            return
        x, y = position or self.plotter.iren.get_event_position()
        if self._picker.Pick(x, y, 0, self.plotter.renderer) and self._picker.GetCellId() >= 0:
            p = self._picker.GetPickPosition()
            self.coords.setText(f"x {p[0]:.4g}   y {p[1]:.4g}   z {p[2]:.4g} {self.length_unit()}")
        else:
            self.coords.setText("")

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
                                  f"space.")

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
        for attr in ("supports_panel", "structural_loads_window"):  # windows on the previous project
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    w.close()
                except RuntimeError:
                    pass
        self.preview_on = False
        self.project, self.path = project, path
        self._saved = self._state() if path else None
        filled = project.complete_settings()                        # a file missing settings: filled, unsaved
        self.init_state()                                           # PyTO-AD's own fresh state
        self._units_to_window()
        for actor in self._rulers:                                  # the old part's scales
            self.plotter.remove_actor(actor, reset_camera=False)
        self._rulers = []
        if os.path.isfile(project.stl_path()):
            loader = ProjectsWindow(self)                           # its load_geometry draws the STL, picking on
            loader.load_geometry(project.stl_path())
            loader.deleteLater()
        else:
            self.plotter.clear_actors()
        self.draw_ruler()
        self.message_text.clear()                                   # the loader's own lines go too
        self.message_text.append(f"Project {project.name}" + (f": {path}" if path else " (not saved yet)"))
        if not os.path.isfile(project.stl_path()):
            self.message_text.append(f"STL not found: {project.stl_path()}")
        if filled:
            self.message_text.append(f"{len(filled)} settings missing from the file were set to their defaults; "
                                     f"Save writes them.")
        self._said = set()
        self._say_checks()
        self._refresh_title()

    def _say_checks(self):
        """The project's errors and warnings in the message panel, each once: a later change repeats none."""
        for line in [f"Error: {e}" for e in self.project.validate()] + \
                    [f"Warning: {w}" for w in self.project.warnings()]:
            if line not in self._said:
                self._said.add(line)
                self.message_text.append(line)

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
            self._say_checks()
            self._refresh_title()

    # ------------------------------------------------------------------ Materials, Formulation, Settings
    def open_panel(self, name):
        """One of PANELS on the project; OK puts its changes in, Cancel drops them."""
        if self.project is None:
            QtWidgets.QMessageBox.information(self, APP, "Start a project first: File > New or File > Open.")
            return None
        panel = PANELS[name](self, self.project)
        if panel.exec_():
            self.apply_panel(panel)
        return panel

    def apply_panel(self, panel):
        lines = panel.apply(self.project)
        for line in lines:
            self.message_text.append(line)
        if lines:
            self._say_checks()
        self._refresh_title()

    def open_materials_panel(self):
        return self.open_panel("Materials")

    def open_formulation_panel(self):
        return self.open_panel("Formulation")

    def open_settings_panel(self):
        return self.open_panel("Settings")

    # ------------------------------------------------------------------ supports and loads
    def open_supports_and_loads_panel(self):
        if self.project is None:
            QtWidgets.QMessageBox.information(self, APP, "Start a project first: File > New or File > Open.")
            return None
        if not self._raise_if_open("supports_panel"):
            self.supports_panel = SupportsPanel(self)
            Strip(self.supports_panel, self, "supports_panel")
        return self.supports_panel

    def open_structural_loads_window(self):
        """The Pick window, docked (in place of MainWindow.open_structural_loads_window)."""
        if not self._raise_if_open("structural_loads_window"):
            self.structural_loads_window = PickWindow(self)
            strip = Strip(self.structural_loads_window, self, "structural_loads_window")
            panel = getattr(self, "supports_panel", None)
            if in_open_strip(panel):                                # in the list's place, as a tab beside it
                self.tabifyDockWidget(panel.parentWidget(), strip)
            strip.raise_()

    def _raise_if_open(self, attr):
        """A docked window that is open, also behind a tab, comes to the front: no second copy."""
        w = getattr(self, attr, None)
        if in_open_strip(w):
            w.parentWidget().raise_()
            return True
        return super()._raise_if_open(attr)

    def project_changed(self):
        self._refresh_title()

    def update_LivVar(self, key, value=True):
        """PyTO-AD's state update; after an Apply in its Structural Loads window, the pick is adopted
        into the project (once the Apply has finished)."""
        super().update_LivVar(key, value)
        if self.project is not None and key in ("structural_loads.forces_applied",
                                                "structural_loads.fixed_constraints"):
            QtCore.QTimer.singleShot(0, self.adopt_picks)

    def adopt_picks(self):
        """The supports and loads applied in PyTO-AD's Structural Loads window become regions and
        supports and loads of the project, and leave PyTO-AD's lists: the project is the only list."""
        torques = [f for f in self.force_data if f.get("type") == "torque"]
        forces = [f for f in self.force_data if f.get("type", "force_xyz") == "force_xyz"]
        sets = sorted({f.get("load_set", 0) for f in forces} - {0})
        names, before = [], len(self.project.bcs)
        if self.constraint_data or forces:
            try:
                names = self.project.adopt_picks({"constraint_data": list(self.constraint_data),
                                                  "force_data": forces})
            except (NotImplementedError, ValueError) as e:
                self.message_text.append(f"Not added: {e}")
        for bc in self.project.bcs[before:]:                        # say what each pick became
            self.message_text.append(f"Added: {bc_name(bc['kind'])} on {bc['on']} "
                                     f"({bc_value(bc, self.project.units)})")
        if torques:
            self.message_text.append("A torque is not one of LSR MMTO's loads, so it was not added.")
        if sets:
            self.message_text.append(f"Load sets {sets}: LSR MMTO has one load case for now, so these loads "
                                     f"were added to it.")
        self.constraint_data, self.force_data = [], []
        self.constrained_triangles.clear()
        w = getattr(self, "structural_loads_window", None)
        try:
            if w is not None:
                w.redraw_loads()
                w.refresh_applied_list()
        except RuntimeError:                                        # the window was already deleted
            pass
        for name in list(self.plotter.actors):     # a load's label is two actors; the redraw leaves the
            if re.search(r"(force|torque)_text_\d+(-points)?$", str(name)):          # point: a stray dot
                self.plotter.remove_actor(name, reset_camera=False)
        panel = getattr(self, "supports_panel", None)
        if in_open_strip(panel):
            panel.after_change()
            if len(self.project.bcs) > before:                      # the round trip: back to the list,
                if in_open_strip(w):                                # the new entry selected
                    w.close()
                panel.parentWidget().raise_()
                panel.bcs.selectRow(len(self.project.bcs) - 1)
            return names
        if self.preview_on:
            self.draw_supports_and_loads()
        self._refresh_title()
        return names

    def preview_mesh(self):
        """The mesh a run of the project uses, built once and kept until the STL or the dofs change."""
        from pyto_project import make_mesh  # type: ignore
        p = self.project
        key = (os.path.abspath(p.stl_path()), p.n_dof, p.single_layer)
        if key != self._mesh_key:
            self.message_text.append(f"Building the mesh ({p.n_dof} dofs) for the preview...")
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                QtWidgets.QApplication.processEvents()
                self._mesh = make_mesh(p.stl_path(), p.n_dof, p.single_layer)
                self._mesh_key = key
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
        return self._mesh

    def draw_points(self, name, points, colour, size=10):
        import pyvista as pv
        self.plotter.remove_actor(name, reset_camera=False)
        if len(points):
            self.plotter.add_mesh(pv.PolyData(np.asarray(points, dtype=float)), color=colour, point_size=size,
                                  render_points_as_spheres=True, pickable=False, name=name)

    def clear_picks(self):
        """Faces and nodes picked but not applied: unselected, their markers gone."""
        self.selected_points, self.selected_regions = [], []
        self.update_point_markers()
        if self.stl_geom:
            self.stl_geom.tri_highlight = [False] * len(self.stl_geom.tri_highlight)
            self.update_highlights()

    def clear_preview(self):
        for name in [n for n in list(self.plotter.actors) if str(n).startswith("mmto_")]:
            self.plotter.remove_actor(name, reset_camera=False)
        self.preview_on = False
        self.plotter.render()

    def draw_supports_and_loads(self):
        """Every region on the run's mesh: supports blue, loads red with an arrow each, other regions
        orange, keep-solid elements grey. Returns (report lines, {region: node count})."""
        import pyvista as pv
        from project import BC_KINDS  # type: ignore
        p, mesh = self.project, self.preview_mesh()
        self.clear_preview()
        xyz, centres = np.asarray(mesh.node_xyz, dtype=float), np.asarray(mesh.elem_centers, dtype=float)
        role = {}
        for bc in p.bcs:
            if bc.get("kind") in BC_KINDS:
                role.setdefault(bc["on"], set()).add(BC_KINDS[bc["kind"]].role)
        lines, counts = [], {}
        for name, s in p.selections.items():
            nodes = s.evaluate(mesh, "nodes", p.base_dir)
            counts[name] = len(nodes)
            colour = (SUPPORT_COLOUR if "support" in role.get(name, ()) else
                      LOAD_COLOUR if "load" in role.get(name, ()) else OTHER_COLOUR)
            if name in role or name not in p.keep_solid_names:
                self.draw_points(f"mmto_{name}", xyz[nodes], colour)
            if name in p.keep_solid_names:                         # the kept elements, as grey blocks
                elems = s.evaluate(mesh, "elements", p.base_dir)
                if len(elems):
                    h = np.asarray(mesh.elem_size, dtype=float)
                    blocks = pv.PolyData(centres[elems]).glyph(
                        geom=pv.Cube(x_length=h[0], y_length=h[1], z_length=h[2]), scale=False, orient=False)
                    self.plotter.add_mesh(blocks, color=KEEP_COLOUR, pickable=False, name=f"mmto_keep_{name}")
                lines.append(f"Keep solid {name}: {len(elems)} elements")
            if len(nodes) == 0:
                lines.append(f"<span style='color:#b00000'>{name} has no nodes on this mesh</span>")
        size = float(np.linalg.norm(np.ptp(xyz, axis=0)))
        for i, bc in enumerate(p.bcs):
            if bc.get("kind") not in ("force", "traction") or bc.get("on") not in p.selections:
                continue
            d = np.asarray(bc.get("total") or bc.get("direction"), dtype=float)
            nodes = p.selections[bc["on"]].evaluate(mesh, "nodes", p.base_dir)
            if np.linalg.norm(d) == 0 or len(nodes) == 0:
                continue
            d = d / np.linalg.norm(d)
            start = xyz[nodes].mean(axis=0) - 0.15 * size * d
            self.plotter.add_mesh(pv.Arrow(start, d, scale=0.15 * size), color=LOAD_COLOUR, pickable=False,
                                  name=f"mmto_arrow{i}")
        try:
            bc_obj, _ = p.supports_and_loads(mesh)
            error = p.rigid_body_error(mesh, bc_obj) if p.bcs else None
            lines.append(f"<span style='color:#b00000'>Error: {error}</span>" if error else
                         "<span style='color:#1a7f37'>The supports hold the part against the loads.</span>"
                         if p.bcs else "No supports or loads yet.")
        except (ValueError, KeyError) as e:
            lines.append(f"<span style='color:#b00000'>Error: {e}</span>")
        lines.insert(0, f"Mesh of the run: {mesh.num_elems} elements, {mesh.num_nodes} nodes")
        self.preview_on = True
        self.plotter.render()
        return lines, counts

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
