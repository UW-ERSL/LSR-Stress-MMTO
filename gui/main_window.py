"""
main_window.py
==============
The LSR MMTO window. It is PyTO-AD's MainWindow (viewer, sidebar, message panel) holding a
Project: File > New, Open, Save and Save As work on .mmto files through Project.load and
Project.save, so a project opened and saved with no edits is written back unchanged (once its
first save has copied into its folder what it uses).

A project lives in a folder of its own (project_folder.py): File > New makes the folder and
copies the STL into it, Save As copies everything into a new folder, and a database or an STL
chosen from elsewhere is copied in. Every save also writes the project's setup script,
<name>_setup.py, again, so it always matches the project. The example projects in projects/ open as templates: Save
asks for a folder of your own, so the examples never change.

The 3D view shows either the part (the Setup view: supports, loads, previews) or a design (the
Design view: a run's result, drawn by draw_design.py). Anything about the setup switches to the
Setup view; the design is kept until the next run, so Show design always brings it back.

    python run_gui.py [project.mmto]        (from the repository folder)

The sidebar keeps PyTO-AD's Units, Geometry and Display Options, and adds Materials,
Formulation and Settings (materials_panel.py, formulation_panel.py, settings_panel.py) and
Supports and loads (supports_and_loads_panel.py). Picking with the
mouse is PyTO-AD's Structural Loads window; each Apply in it is adopted into the project.
"""
import json
import os
import re
import sys

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QKeySequence

from .formulation_panel import FormulationPanel
from .materials_panel import MaterialsPanel
from .panel_widgets import GREY, Info, ok_cancel
from .settings_panel import SettingsPanel
from .supports_and_loads_panel import KEEP_COLOUR, LOAD_COLOUR, OTHER_COLOUR, SUPPORT_COLOUR, SupportsPanel, bc_name, bc_value
from .user_guide_window import open_user_guide
from .pyto_ad_imports import (FILE_DIALOG_OPTIONS, UNIT_SYSTEMS, GeometryWindow, MainWindow, ProjectsWindow,
                      StructuralLoadsWindow, UnitsWindow)

APP = "LSR MMTO"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(REPO, "projects")   # the example projects: opened as templates, never changed
OUTLINE_COLOUR = "#c400c4"                  # selection shapes: magenta, unlike the part, supports, loads
SNAPS = [("Snap: off", 0.0), ("Snap: 0.1", 0.1), ("Snap: 1", 1.0), ("Snap: 5", 5.0), ("Snap: 10", 10.0)]
PROJECT_FILTER = "LSR MMTO project (*.mmto)"
SCRIPT_FILTER = "Python script (*.py)"
ICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "lsr_mmto.ico")
SIDEBAR = ("Units", "Geometry", "Materials", "Supports and loads", "Formulation", "Settings", "Run",
           "Results", "Display Options")
FOLDER_HELP = ("A project keeps everything it uses in its own folder: the project file, the part (in part), "
               "the material database and its trained network (in materials), and the results of every run "
               "(in results). Zip the folder to send the whole project.")
PANELS = {"Materials": MaterialsPanel, "Formulation": FormulationPanel, "Settings": SettingsPanel}


def point_text(p):
    """A point as a field takes it: 120, 80, 1."""
    return ", ".join(format(float(v), ".6g") for v in p)


def load_direction(bc):
    """The arrow of a load: its kind's first vector value (a force's total, a pressure's direction), or None
    for a support or a load with no vector."""
    from lsr_mmto.problem.boundary_conditions import BC_KINDS
    kind = BC_KINDS.get(bc.get("kind"))
    if kind is None or kind.role != "load":
        return None
    v = next((bc.get(name) for name, typ, _, _ in kind.fields if typ == "vector"), None)
    return None if v is None else np.asarray(v, dtype=float)


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
        if getattr(dialog, "optimization_running", False):        # a run going: its strip stays (Stop first)
            QtWidgets.QMessageBox.information(self, "Run", "A run is going. Stop it first: the Stop button "
                                                           "keeps the design reached so far.")
            event.ignore()
            return
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


def place_outline(selection, mesh):
    """A solid showing a selection's shape (drawn as its edges), or None for one with no shape of its own (picked
    faces, rows, a node, a list of numbers). selection has every selection(name) in it resolved."""
    import pyvista as pv
    from lsr_mmto.problem import selections as S
    xyz = np.asarray(mesh.node_xyz, dtype=float)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    h = 0.5 * np.asarray(mesh.elem_size, dtype=float)
    kind = selection.TYPE
    if kind in ("union", "intersect", "difference"):
        parts = [o for o in (place_outline(s, mesh) for s in selection.parts) if o is not None]
        return parts[0].merge(parts[1:]) if len(parts) > 1 else (parts[0] if parts else None)
    if kind == "sphere":
        return pv.Sphere(radius=S._length(mesh, selection.radius), center=selection.center,
                         theta_resolution=12, phi_resolution=8)              # a light wire ball
    if kind == "cylinder":
        axis = S._axis_vector(selection.axis)
        c = np.asarray(selection.center, dtype=float)
        if selection.along is not None:
            t0, t1 = (S._length(mesh, v) for v in selection.along)
        else:                                                       # through the whole part
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            t = (corners - c) @ axis
            t0, t1 = t.min(), t.max()
        out = None
        for r in (S._length(mesh, selection.radius), S._length(mesh, selection.inner_radius)):
            if r > 0:
                tube = pv.Cylinder(center=c + axis * (t0 + t1) / 2, direction=axis, radius=r, height=t1 - t0,
                                   resolution=48, capping=False)
                out = tube if out is None else out.merge(tube)
        return out
    a, b = lo - h, hi + h                                           # slabs and boxes, a little proud of the part
    if kind == "box":
        a, b = np.asarray(selection.lo, dtype=float), np.asarray(selection.hi, dtype=float)
        thin = b - a < 2 * h
        a, b = np.where(thin, a - h, a), np.where(thin, b + h, b)
    elif kind == "plane":
        k = S._AXES[selection.axis]
        a[k], b[k] = selection.value - h[k], selection.value + h[k]
        for axis, (l, u) in (selection.limits or {}).items():
            j = S._AXES[axis]
            a[j], b[j] = l - h[j], u + h[j]
    elif kind in ("face", "near_face"):
        k, end = S._side(selection.side)
        edge = lo[k] if end == "min" else hi[k]
        depth = S._length(mesh, selection.distance) if kind == "near_face" else h[k]
        a[k], b[k] = (edge - h[k], edge + depth) if end == "min" else (edge - depth, edge + h[k])
    else:
        return None
    return pv.Box(bounds=(a[0], b[0], a[1], b[1], a[2], b[2]))


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


def is_example(path):
    """True for one of the example projects that come with LSR MMTO (projects/)."""
    return path is not None and os.path.normcase(os.path.dirname(os.path.abspath(path))) == \
        os.path.normcase(os.path.abspath(EXAMPLES))


class ProjectFolderDialog(QtWidgets.QDialog):
    """Name and location of a new project folder, and for a new project its part (STL)."""

    def __init__(self, parent, title, location, name="", stl=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(560, 0)
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.name = QtWidgets.QLineEdit(name)
        self.location = QtWidgets.QLineEdit(location)
        form.addRow("Name", self.name)
        form.addRow("Location", self._with_browse(self.location, self._browse_location))
        self.stl = None
        if stl is not None:
            self.stl = QtWidgets.QLineEdit(stl)
            form.addRow("Part (STL)", self._with_browse(self.stl, self._browse_stl))
        layout.addLayout(form)
        row = QtWidgets.QHBoxLayout()
        self.where = QtWidgets.QLabel("")
        self.where.setStyleSheet(GREY)
        self.where.setWordWrap(True)
        row.addWidget(self.where, stretch=1)
        row.addWidget(Info(FOLDER_HELP))
        layout.addLayout(row)
        for edit in (self.name, self.location):
            edit.textChanged.connect(self._show_where)
        self._show_where()
        ok_cancel(self, layout)

    @staticmethod
    def _with_browse(edit, slot):
        box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, stretch=1)
        button = QtWidgets.QPushButton("Browse...")
        button.clicked.connect(slot)
        row.addWidget(button)
        return box

    def _browse_location(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Location", self.location.text(),
                                                            options=FILE_DIALOG_OPTIONS)
        if folder:
            self.location.setText(folder)

    def _browse_stl(self):
        stl, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Part (STL)", self.location.text(),
                                                       "STL Files (*.stl *.STL)", options=FILE_DIALOG_OPTIONS)
        if stl:
            self.stl.setText(stl)
            if not self.name.text().strip():
                self.name.setText(os.path.splitext(os.path.basename(stl))[0])

    def folder(self):
        return os.path.join(self.location.text().strip(), self.name.text().strip())

    def stl_path(self):
        return self.stl.text().strip() if self.stl is not None else None

    def _show_where(self):
        self.where.setText(f"The project folder: {self.folder()}" if self.name.text().strip() else "")

    def problems(self):
        from lsr_mmto.problem.project_folder import folder_problem, name_problem
        out = []
        bad_name = name_problem(self.name.text())                   # before joining: on Windows "a:b" joined
        if bad_name:                                                # to a folder is a path on drive A:
            out.append(f"Name: {bad_name}")
        if not os.path.isdir(self.location.text().strip()):
            out.append("Location: no such folder")
        elif not bad_name:
            problem = folder_problem(self.folder())
            if problem:
                out.append(f"Name: {problem}")
        if self.stl is not None and not os.path.isfile(self.stl_path()):
            out.append("Part (STL): choose the STL file")
        return out

    def accept(self):
        problems = self.problems()
        if problems:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(), "\n".join(problems))
            return
        super().accept()


class MMTOWindow(MainWindow):
    """PyTO-AD's MainWindow with a Project, a File menu, and the sidebar of SIDEBAR: PyTO-AD's
    buttons where it has them, ours (PANELS) for the rest."""

    BUTTON_CONFIG = [next((c for c in MainWindow.BUTTON_CONFIG if c["name"] == name),
                          {"name": name, "icon": "arrow", "always_enabled": True,
                           "handler": f"open_{name.lower().replace(' ', '_')}_panel"}) for name in SIDEBAR]
    template = False                        # an example project: Save asks for a folder of one's own
    view = "setup"                          # what the 3D view shows: "setup" (the part) or "design"
    _design = None                          # draws the design kept for the Design view, or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project, self.path, self._saved = None, None, None
        self._mesh, self._mesh_key, self.preview_on = None, None, False
        self.design_bars, self._view_buttons = set(), []            # colour scales of the design; Show buttons
        self._said = set()                                          # errors and warnings already shown
        self.message_text.setText(f"{APP}. File > New starts a project from an STL; File > Open opens a "
                                  f".mmto project.")
        self._build_menu()
        self._refresh_title()

    # ------------------------------------------------------------------ the window
    def create_status_bar(self):
        """The part's extent on the left, with its facts behind an (i); on the right, the point under the
        mouse and the Ruler tick."""
        self.extent = QtWidgets.QLabel(APP)                        # the part's extent, once there is one
        self.statusBar().addWidget(self.extent)
        self.facts = Info("")                                       # the part, its volume and size, the database
        self.facts.setVisible(False)
        self.statusBar().addWidget(self.facts)
        self.coords = QtWidgets.QLabel("")
        self.coords.setStyleSheet("color: #666666;")
        self.coords.setMinimumWidth(240)
        self.ruler_box = QtWidgets.QCheckBox("Ruler")
        self.ruler_box.setToolTip("Show scales along the part, in the project's length unit")
        self.ruler_box.toggled.connect(lambda on: self.draw_ruler())
        self.measure = QtWidgets.QPushButton("Measure")
        self.measure.setCheckable(True)
        self.measure.setToolTip("Click two points on the part to see the distance between them; Esc to stop")
        self.measure.toggled.connect(self._measure_toggled)
        self.measured = QtWidgets.QLabel("")
        self.snap = QtWidgets.QComboBox()
        for text, step in SNAPS:
            self.snap.addItem(text, step)
        self.snap.setToolTip("Round the point under the mouse, a copied point and Measure to this step, in the "
                             "project's length unit")
        self._measure_points = []
        self._measure_esc = QtWidgets.QShortcut(QKeySequence(QtCore.Qt.Key_Escape), self,
                                                activated=lambda: self.measure.setChecked(False))
        self._measure_esc.setEnabled(False)                         # Esc stops measuring, and only that
        self.statusBar().addPermanentWidget(self.measured)
        self.statusBar().addPermanentWidget(self.coords)
        self.statusBar().addPermanentWidget(self.snap)
        self.statusBar().addPermanentWidget(self.measure)
        self.statusBar().addPermanentWidget(self.ruler_box)
        self._rulers = []
        self._picker = None
        QtCore.QTimer.singleShot(0, self._track_mouse)              # once the 3D view exists

    # ------------------------------------------------------------------ scale: the ruler and the mouse
    def length_unit(self):
        from .supports_and_loads_panel import field_unit
        return field_unit("length", self.project.units) if self.project is not None else ""

    def part_bounds(self):
        """(lowest, highest) corner of the part's STL, or None with no part."""
        if not getattr(self, "stl_geom", None):
            return None
        p = np.asarray(self.stl_geom.mesh.vectors, dtype=float).reshape(-1, 3)
        return p.min(axis=0), p.max(axis=0)

    @staticmethod
    def length_format(size):
        """printf format for lengths on a part of this size: 200 mm as 200, 0.0285 m as 0.029."""
        if size >= 20:
            return "%.0f"
        if size >= 1:
            return "%.1f"
        return f"%.{int(np.ceil(-np.log10(size))) + 1}f" if size > 0 else "%g"

    def part_extent_text(self):
        """The part's extent in words, for the region form: x 0 to 200, y 0 to 200, z 0 to 1 mm."""
        b = self.part_bounds()
        if b is None:
            return ""
        lo, hi = b
        f = self.length_format(float(np.max(hi - lo)))
        parts = ", ".join(f"{a} {f % lo[k]} to {f % hi[k]}" for k, a in enumerate("xyz"))
        return f"The part spans {parts} {self.length_unit()}".rstrip() + "."

    def part_facts(self):
        """The part's facts, shown behind the (i) beside its extent."""
        b = self.part_bounds()
        if b is None or self.project is None:
            return ""
        unit = self.length_unit()
        size = b[1] - b[0]
        f = self.length_format(float(np.max(size)))
        lines = [f"Part: {os.path.basename(self.project.stl_path())}"]
        try:
            volume = self.stl_geom.compute_mass_properties()[1]
            lines.append(f"Volume: {volume:.4g} {unit}³".rstrip())
        except Exception:                                           # noqa: BLE001
            pass
        lines.append("Size: " + " × ".join(f % v for v in size) + f" {unit}".rstrip())
        lines.append(f"Material database: {self.project.database or 'none chosen yet'}")
        return "\n".join(lines)

    def update_part_facts(self):
        text = self.part_facts()
        self.facts.setToolTip(text)
        self.facts.setVisible(bool(text))

    def update_geometry_info_text(self):
        """In place of PyTO-AD's text over the 3D view (model, volume, length, material): the part's facts
        go behind the (i) in the status bar, clear of the view."""
        self.plotter.remove_actor("geometry_info", reset_camera=False)
        self.update_part_facts()

    def draw_ruler(self):
        """Scales along the part's lowest x and y edges (and z, when the part has thickness), on or off
        with the Ruler tick."""
        for actor in self._rulers:
            self.plotter.remove_actor(actor, reset_camera=False)
        for name in [n for n in self.plotter.actors if str(n).startswith("ruler_origin")]:  # its dot too
            self.plotter.remove_actor(name, reset_camera=False)
        self._rulers = []
        b = self.part_bounds()
        self.coords.setText("")                                     # a new part: no old reading
        self._index_part()
        self.extent.setText(APP if b is None else
                            "Part: " + self.part_extent_text()[len("The part spans "):-1].replace(", ", " \u00b7 "))
        self.update_part_facts()
        if b is not None and self.ruler_box.isChecked():
            lo, hi = b
            size = float(np.max(hi - lo))
            off = 0.06 * size
            style = dict(number_labels=3, font_size_factor=0.6, label_size_factor=1.3, tick_length=3,
                         tick_label_offset=6, label_color="#8a8a8a", tick_color="#b5b5b5",
                         label_format=self.length_format(size))
            x0, y0, z1 = lo[0] - off, lo[1] - off, hi[2]
            lines = [([lo[0], y0, z1], [hi[0], y0, z1], False, self.length_unit()),
                     ([x0, hi[1], z1], [x0, lo[1], z1], True, "")]
            if hi[2] - lo[2] > 0.05 * size:                         # a plate shows no z scale; at the far
                x1, y1 = hi[0] + off, hi[1] + off                   # corner, clear of x and y when seen head on
                lines.append(([x1, y1, lo[2]], [x1, y1, hi[2]], False, ""))
            for a, b_, flip, title in lines:
                actor = self.plotter.add_ruler(a, b_, flip_range=flip, title=title, **style)
                for prop in (actor.GetLabelTextProperty(), actor.GetTitleTextProperty()):
                    prop.ItalicOff()
                    prop.BoldOff()
                    prop.ShadowOff()
                actor.SetPickable(False)
                self._rulers.append(actor)
            self._rulers.append(self.plotter.add_point_labels(          # the origin, where 0, 0, 0 is
                [[0.0, 0.0, 0.0]], ["0"], point_size=9, font_size=11, text_color="#666666",
                point_color="#666666", shape=None, render_points_as_spheres=True, always_visible=True,
                pickable=False, name="ruler_origin", reset_camera=False))
        self.plotter.render()

    def _track_mouse(self):
        """The point of the part under the mouse, in the status bar. A mouse move only notes where the
        mouse is; the lookup runs at most every 50 ms, never while a button is held (rotating, zooming)
        and never on top of a lookup still running, so a large part cannot pile them up."""
        import vtk
        try:
            self._picker = vtk.vtkCellPicker()
            self._picker.SetTolerance(0.0005)
            self._mouse_at, self._looking = None, False
            self._mouse_timer = QtCore.QTimer(self)
            self._mouse_timer.setSingleShot(True)
            self._mouse_timer.timeout.connect(self._look_up_mouse)
            self.plotter.iren.add_observer("MouseMoveEvent", self._mouse_moved)
        except Exception:                                           # noqa: BLE001  (no 3D view: no readout)
            self._picker = None

    def _mouse_moved(self, *_):
        self._mouse_at = self.plotter.iren.get_event_position()
        if not self._mouse_timer.isActive():
            self._mouse_timer.start(50)

    def _look_up_mouse(self):
        if self._looking or QtWidgets.QApplication.mouseButtons() != QtCore.Qt.NoButton:
            return                                                  # rotating: read it when the button is up
        self._looking = True
        try:
            self.show_point_under_mouse(self._mouse_at)
        finally:
            self._looking = False

    def part_actor(self):
        """The part's own actor in the 3D view (its STL), or None."""
        n = len(self.stl_geom.mesh.vectors) if getattr(self, "stl_geom", None) else -1
        for actor in self.plotter.renderer.GetActors():
            data = actor.GetMapper().GetInput() if actor.GetMapper() else None
            if data is not None and data.GetNumberOfCells() == n:
                return actor
        return None

    def _index_part(self):
        """The lookup searches only the part, through an index of its triangles built once per part."""
        import vtk
        if self._picker is None:
            return
        self._picker.InitializePickList()
        self._picker.RemoveAllLocators()
        actor = self.part_actor()
        if actor is not None:
            locator = vtk.vtkStaticCellLocator()
            locator.SetDataSet(actor.GetMapper().GetInput())
            locator.BuildLocator()
            self._picker.AddLocator(locator)
            self._picker.AddPickList(actor)
        self._picker.SetPickFromList(True)

    def show_point_under_mouse(self, position=None):
        if self._picker is None or self.part_bounds() is None or position is None and self._mouse_at is None:
            self.coords.setText("")
            return
        x, y = position or self._mouse_at
        if self._picker.Pick(x, y, 0, self.plotter.renderer) and self._picker.GetCellId() >= 0:
            p = np.asarray(self._picker.GetPickPosition())
            self.coords.setText(f"x {p[0]:.4g}   y {p[1]:.4g}   z {p[2]:.4g}" + (
                f"  \u2192  {point_text(self.snapped(p))}" if self.snap.currentData() else "") +
                f" {self.length_unit()}")
        else:
            self.coords.setText("")
        panel = getattr(self, "supports_panel", None)
        if in_open_strip(panel):                                    # also on the form in front, if any
            panel.show_mouse(self.coords.text())

    def snapped(self, p):
        """p rounded to the Snap step (unchanged with Snap off)."""
        step = self.snap.currentData() or 0.0
        p = np.asarray(p, dtype=float)
        return np.round(p / step) * step if step else p

    # ------------------------------------------------------------------ measuring
    def _measure_toggled(self, on):
        self._measure_esc.setEnabled(on)
        self._measure_points = []
        self._draw_measure()
        self.measured.setText("Measure: click the first point" if on else "")

    def on_left_button_press(self, picked_point, picker=None):
        """A click on the part, by what is in front: Measure on, a point to measure; the Pick strip open,
        picking; a form open under Supports and loads, the (snapped) point copied to paste; else picking."""
        panel = getattr(self, "supports_panel", None)
        if not self.measure.isChecked():
            form_open = in_open_strip(panel) and panel.stack.currentIndex() > 0
            if form_open and not in_open_strip(getattr(self, "structural_loads_window", None)):
                if picker is not None and picker.GetCellId() >= 0:
                    text = point_text(self.snapped(picked_point))
                    QtWidgets.QApplication.clipboard().setText(text)
                    panel.show_copied(text)
                return None
            return super().on_left_button_press(picked_point, picker)
        if picker is None or picker.GetCellId() < 0:
            return
        if len(self._measure_points) == 2:                          # a third click starts again
            self._measure_points = []
        self._measure_points.append(self.snapped(picked_point))
        self._draw_measure()
        if len(self._measure_points) == 1:
            self.measured.setText("Measure: click the second point")
        else:
            a, b = self._measure_points
            d, unit, f = b - a, self.length_unit(), "%.4g"
            self.measured.setText(f"Distance {f % np.linalg.norm(d)} {unit}  (dx {f % d[0]}, dy {f % d[1]}, "
                                  f"dz {f % d[2]})")

    def _draw_measure(self):
        import pyvista as pv
        for name in [n for n in self.plotter.actors if str(n).startswith("measure_")]:
            self.plotter.remove_actor(name, reset_camera=False)
        pts = self._measure_points
        if pts:
            self.plotter.add_mesh(pv.PolyData(np.vstack(pts)), color="#222222", point_size=12,
                                  render_points_as_spheres=True, pickable=False, name="measure_points")
        if len(pts) == 2:
            self.plotter.add_mesh(pv.Line(pts[0], pts[1]), color="#222222", line_width=3, pickable=False,
                                  name="measure_line")
        self.plotter.render()

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
        file_menu.addAction("&Export as script").triggered.connect(lambda: self.export_script())
        file_menu.addAction("&Run script...").triggered.connect(lambda: self.run_script())
        file_menu.addSeparator()
        file_menu.addAction("E&xit").triggered.connect(self.close)
        help_menu = self.menuBar().addMenu("&Help")
        guide = help_menu.addAction("&User guide")
        guide.setShortcut(QKeySequence.HelpContents)                # F1
        guide.triggered.connect(lambda: open_user_guide(self))
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
        example = " (example)" if self.template else ""
        self.setWindowTitle(f"{name}{example}{'*' if self.is_dirty() else ''} - {APP}")

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
        if not self.run_going() and self._ask_to_save():
            event.accept()
        else:
            event.ignore()

    # ------------------------------------------------------------------ New, Open, Save
    def _folder(self):
        return os.path.dirname(self.path) if self.path else os.getcwd()

    def _location(self):
        """Where a new project folder goes by default: beside the project open now (in its own folder),
        or else the Documents folder."""
        from lsr_mmto.problem.project_folder import has_own_folder
        if self.project is not None and self.path and not self.template:
            here = os.path.dirname(self.path)
            return os.path.dirname(here) if has_own_folder(self.project) else here
        documents = os.path.join(os.path.expanduser("~"), "Documents")
        return documents if os.path.isdir(documents) else os.path.expanduser("~")

    def own_folder(self):
        """True when the project open is saved in a project folder of its own (not an example)."""
        from lsr_mmto.problem.project_folder import has_own_folder
        return self.project is not None and self.path is not None and not self.template and \
            has_own_folder(self.project)

    def run_going(self):
        """True, after saying so, when a run is going: no other project, no closing, until it stops."""
        run = getattr(self, "run_window", None)
        if in_open_strip(run) and run.optimization_running:
            QtWidgets.QMessageBox.information(self, APP, "A run is going. Stop it first (Run > Stop); the design "
                                                         "reached so far is kept.")
            return True
        return False

    def new_project(self, stl=None, folder=None):
        """A project in a new folder of its own, every setting at its default: the folder is made, the STL
        copied into it and the project saved there. With no folder given, a dialog asks for the name,
        the location and the STL."""
        from lsr_mmto.problem import project_folder
        if self.run_going() or not self._ask_to_save():
            return False
        if folder is None or stl is None:
            dialog = ProjectFolderDialog(self, "New project", self._location(), stl=stl or "")
            if not dialog.exec_():
                return False
            folder, stl = dialog.folder(), dialog.stl_path()
        try:
            p, path = project_folder.create(folder, stl)
            project_folder.write_setup_script(p)
        except (OSError, ValueError) as e:
            QtWidgets.QMessageBox.warning(self, "New project", f"The project was not made: {e}")
            return False
        self._show(p, path)
        return True

    def open_project(self, path=None):
        from lsr_mmto.problem.project import Project
        if self.run_going() or not self._ask_to_save():
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
        """Save in place; a project never saved, or an example, is saved as a new project (Save As)."""
        if self.project is None:
            return False
        if self.path is None or self.template:
            return self.save_project_as()
        return self._write(self.path)

    def save_project_as(self, path=None):
        """Save as a new project: with no path, a dialog asks for a name and a location and makes the
        project folder there. Everything the project uses (the STL, the database and its network) is
        copied into the folder, and the project takes the new name. Results stay with the old project."""
        from lsr_mmto.problem import project_folder
        if self.project is None:
            return False
        if path is None:
            dialog = ProjectFolderDialog(self, "Save as a new project", self._location(),
                                         name=self.project.name if self.template else f"{self.project.name} copy")
            if not dialog.exec_():
                return False
            path = os.path.join(dialog.folder(), f"{os.path.basename(dialog.folder())}.mmto")
        if not path.lower().endswith(".mmto"):
            path += ".mmto"
        packed = self.project.copy(name=os.path.splitext(os.path.basename(path))[0])
        try:
            project_folder.pack(packed, os.path.dirname(os.path.abspath(path)))
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, "Save project", f"Not saved: {e}")
            return False
        p = self.project
        old = p.name, p.base_dir, p.stl, p.database, p.selections
        # the regions too: pack points a region's own files (an inside_stl surface) at their copies in part/
        p.name, p.base_dir, p.stl, p.database, p.selections = (packed.name, packed.base_dir, packed.stl,
                                                              packed.database, packed.selections)
        if not self._write(path):
            p.name, p.base_dir, p.stl, p.database, p.selections = old
            return False
        if self.template:
            self.template = False
            self.message_text.append("Saved as a project of your own; the example is unchanged.")
            self._refresh_title()
        self.update_part_facts()
        return True

    # ------------------------------------------------------------------ the project as a Python script
    def export_script(self):
        """The project's setup script, <name>_setup.py in its folder (project_folder.setup_script), written
        again from the project after a save, so that it matches the file. Every save writes it too; this
        says where it is. Edited in any editor, it comes back with Run script."""
        from lsr_mmto.problem import project_folder
        if self.project is None:
            return None
        if self.path is None or self.template or self.is_dirty():
            answer = QtWidgets.QMessageBox.question(
                self, "Export as script", "The project is saved first, so the script matches the file. Save now?",
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Cancel)
            if answer != QtWidgets.QMessageBox.Save or not self.save_project():
                return None
        try:
            path = project_folder.write_setup_script(self.project)
        except (OSError, ValueError) as e:
            QtWidgets.QMessageBox.warning(self, "Export as script", f"Not written: {e}")
            return None
        self.message_text.append(f"Script written: {path}. Edit it in any editor, then File > Run script.")
        return path

    def run_script(self, path=None):
        """A Python script run with this Python, in the background; what it prints goes to the message
        panel, and the project it saves (the newest .mmto written in its folder while it ran) is opened.
        Unsaved changes are asked about first, since the script may save over the project file."""
        if self.run_going() or not self._ask_to_save():
            return False
        if getattr(self, "script_process", None) is not None:
            QtWidgets.QMessageBox.information(self, "Run script", "A script is running; wait for it to finish.")
            return False
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Run script", self._folder(), SCRIPT_FILTER,
                                                            options=FILE_DIALOG_OPTIONS)
            if not path:
                return False
        import time
        path = os.path.abspath(path)
        folder = os.path.dirname(path)
        process = QtCore.QProcess(self)
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.setWorkingDirectory(folder)
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")                    # what it prints, read the same on Windows
        process.setProcessEnvironment(env)
        self._script_text = ""
        process.readyReadStandardOutput.connect(lambda: self._script_output(process))
        started = time.time()
        process.finished.connect(lambda code, _status: self._script_finished(path, code, started))
        self.script_process = process
        self.message_text.append(f"Running {path}...")
        process.start(sys.executable, ["-u", path])
        if not process.waitForStarted(10000):
            self.script_process = None
            self.message_text.append(f"The script could not be started: {process.errorString()}")
            return False
        return True

    def _script_output(self, process, final=False):
        """What the script printed, a whole line at a time, into the message panel."""
        self._script_text += bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        lines = self._script_text.replace("\r\n", "\n").split("\n")
        self._script_text = "" if final else lines.pop()
        for line in lines:
            if line.strip():
                self.message_text.append(line)

    def _script_finished(self, path, code, started):
        self._script_output(self.script_process, final=True)
        self.script_process = None
        name = os.path.basename(path)
        if code != 0:
            self.message_text.append(f"{name} stopped with an error; what it printed is above.")
            return
        folder = os.path.dirname(path)
        saved = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".mmto")]
        saved = [f for f in saved if os.path.getmtime(f) >= started]
        if not saved:
            self.message_text.append(f"{name} ran, but saved no project in its folder (a script ends with "
                                     f"p.save(...)).")
            return
        newest = max(saved, key=os.path.getmtime)
        if self.project is not None:
            self._saved = self._state()                             # asked about already: no second question
        if self.open_project(newest):
            self.message_text.append(f"{name} ran: {newest} opened.")

    def _write(self, path):
        """The project saved to path, and its setup script written again to match (the window's saves only)."""
        from lsr_mmto.problem.project_folder import write_setup_script
        try:
            self.project.save(path)
        except Exception as e:                                      # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Save project", f"Not saved: {e}")
            return False
        self.path, self._saved = os.path.abspath(path), self._state()
        self.message_text.append(f"Saved {self.path}")
        try:
            write_setup_script(self.project)
        except (OSError, ValueError) as e:
            self.message_text.append(f"The setup script was not written: {e}")
        self._refresh_title()
        return True

    def _show(self, project, path):
        """Make project the one in the window: PyTO-AD's state reset, units synced, the STL drawn."""
        for attr in ("supports_panel", "structural_loads_window", "run_window", "results_window"):
            w = getattr(self, attr, None)                           # windows on the previous project
            if w is not None:
                try:
                    w.close()
                except RuntimeError:
                    pass
        self.preview_on = False
        self.project, self.path = project, path
        self.template = is_example(path)
        self._design, self.view, self._shown_wide = None, "setup", False    # the previous project's design goes
        self.design_bars.clear()
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
            self.plotter.remove_actor("geometry_info", reset_camera=False)    # its text over the view: the
        else:                                                                # facts are behind the (i)
            self.plotter.clear_actors()
        self.draw_ruler()
        self._view_changed()
        self.message_text.clear()                                   # the loader's own lines go too
        self.message_text.append(f"Project {project.name}" + (f": {path}" if path else " (not saved yet)"))
        if self.template:
            self.message_text.append("This is an example. Save makes a copy in a project folder of your own; the "
                                     "example itself stays as it is.")
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
        if lines and isinstance(panel, MaterialsPanel) and self.own_folder():
            from lsr_mmto.problem.project_folder import bring_in_database
            try:
                copied = bring_in_database(self.project)
            except OSError as e:
                copied, lines = None, lines + [f"The database could not be copied into the project: {e}"]
            if copied:
                lines.append(f"Copied into the project folder: {copied}, with its trained network if it has one")
        self.update_part_facts()
        for line in lines:
            self.message_text.append(line)
        if lines:
            self._say_checks()
        self._refresh_title()

    def database_folder(self):
        """Where New database... writes by default: the project's materials folder; for an example,
        not beside it (the examples never change)."""
        if self.own_folder():
            folder = os.path.join(self.project.base_dir, "materials")
            os.makedirs(folder, exist_ok=True)
            return folder
        return self._location() if self.template else self.project.base_dir

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
        self.show_setup_view()
        if not self._raise_if_open("supports_panel"):
            self.supports_panel = SupportsPanel(self)
            Strip(self.supports_panel, self, "supports_panel")
        return self.supports_panel

    def open_run_panel(self):
        """The Run strip (PyTO-AD's Optimize window on LSR MMTO's run), beside Supports and loads."""
        if self.project is None:
            QtWidgets.QMessageBox.information(self, APP, "Start a project first: File > New or File > Open.")
            return None
        if not self._raise_if_open("run_window"):
            from .run_panel import RunWindow
            self.run_window = RunWindow(self)
            self._dock_beside(Strip(self.run_window, self, "run_window"))
        return self.run_window

    def open_results_panel(self, folder=None):
        """The Results strip: this project's runs, each shown on the part and in tabs; folder, a run
        to show at once."""
        if self.project is None:
            QtWidgets.QMessageBox.information(self, APP, "Start a project first: File > New or File > Open.")
            return None
        if not self._raise_if_open("results_window"):
            from .results_panel import ResultsWindow
            self.results_window = ResultsWindow(self)
            self._dock_beside(Strip(self.results_window, self, "results_window"))
        else:
            self.results_window.refresh()
        if folder is not None:
            self.results_window.select(folder)
        return self.results_window

    def _dock_beside(self, strip):
        """A new strip as a tab beside the strips already open, in front."""
        for attr in ("supports_panel", "run_window", "results_window"):
            other = getattr(self, attr, None)
            if other is not strip.widget() and in_open_strip(other):
                self.tabifyDockWidget(other.parentWidget(), strip)
                break
        strip.raise_()

    def open_structural_loads_window(self):
        """The Pick window, docked (in place of MainWindow.open_structural_loads_window)."""
        self.show_setup_view()
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
        from lsr_mmto.problem.mesh import make_mesh
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

    def _remove_preview(self):
        for name in [n for n in list(self.plotter.actors) if str(n).startswith("mmto_")]:
            self.plotter.remove_actor(name, reset_camera=False)
        self.preview_on = False

    def clear_preview(self):
        """The preview drawings gone, back to the Setup view. The design is kept (Show design)."""
        self._remove_preview()
        self.show_setup_view(render=False)
        self.plotter.render()

    # ------------------------------------------------------------------ Setup view and Design view
    def set_design(self, draw, show=True, wide=False):
        """draw(window) draws a design (design.py); it is kept for the Design view until another is set
        (the next run) or the project changes. Shown now when show, or when the Design view is on.
        wide: it reaches beyond the part (two designs side by side), so the view is fitted to it."""
        self._design, self._wide = draw, wide
        if show or self.view == "design":
            self.show_design_view()
        else:
            self._view_changed()

    def _remove_design(self):
        from .draw_design import PREFIX
        for name in [n for n in list(self.plotter.actors) if str(n).startswith(PREFIX)]:
            self.plotter.remove_actor(name, reset_camera=False)
        for title in list(self.design_bars):
            try:
                self.plotter.remove_scalar_bar(title, render=False)
            except (KeyError, ValueError, TypeError):
                pass
        self.design_bars.clear()

    def show_design_view(self):
        """The design in place of the part; previews are cleared. False when there is no design."""
        if self._design is None:
            return False
        self._remove_design()
        self._remove_preview()
        part = self.part_actor()
        if part is not None:
            part.SetVisibility(False)
        self.view = "design"
        try:
            self._design(self)
        except Exception as e:                                      # noqa: BLE001
            self.message_text.append(f"The design could not be drawn: {e}")
        wide = getattr(self, "_wide", False)
        if wide or getattr(self, "_shown_wide", False):             # into or out of two designs: fit the view
            self.plotter.reset_camera()
        self._shown_wide = wide
        self._view_changed()
        self.plotter.render()
        return True

    def show_setup_view(self, render=True):
        """The part, for its supports, loads and previews; the design is kept for Show design."""
        if self.view == "setup":
            return
        self._remove_design()
        part = self.part_actor()
        if part is not None:
            part.SetVisibility(True)
        if getattr(self, "_shown_wide", False):                     # back from two designs: the part fills
            self.plotter.reset_camera()                             # the view again
            self._shown_wide = False
        self.view = "setup"
        self._view_changed()
        if render:
            self.plotter.render()

    def toggle_view(self):
        if self.view == "design":
            self.show_setup_view()
        else:
            self.show_design_view()

    def view_button(self):
        """A Show setup / Show design button that follows the view; shown while there is a design."""
        button = QtWidgets.QPushButton("Show design")
        button.clicked.connect(self.toggle_view)
        self._view_buttons.append(button)
        self._view_changed()
        return button

    def _view_changed(self):
        for button in list(self._view_buttons):
            try:
                button.setVisible(self._design is not None)
                button.setText("Show setup" if self.view == "design" else "Show design")
            except RuntimeError:                                    # its strip was closed
                self._view_buttons.remove(button)

    def _arrow(self, i, direction, points, size):
        """A load's arrow, its tip on the load's nodes, coming from outside the part. A load that pulls
        away from the part would put that arrow inside the part, hidden: it starts on the nodes instead."""
        import pyvista as pv
        d = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
        at, length = np.asarray(points).mean(axis=0), 0.15 * size
        start = at - length * d
        if self._inside_part(at - 0.5 * length * d):                # where the arrow would be: inside the part
            start = at
        self.plotter.add_mesh(pv.Arrow(start, d, scale=length), color=LOAD_COLOUR, pickable=False,
                              name=f"mmto_arrow{i}")

    def _inside_part(self, point):
        """True when point is inside an element of the preview mesh (within half an element's diagonal of
        an element centre)."""
        mesh = self._mesh
        if mesh is None:
            return False
        if getattr(self, "_centre_tree", (None,))[0] is not mesh:
            from scipy.spatial import cKDTree
            self._centre_tree = (mesh, cKDTree(np.asarray(mesh.elem_centers, dtype=float)))
        distance, _ = self._centre_tree[1].query(np.asarray(point, dtype=float))
        return distance <= 0.5 * float(np.linalg.norm(np.asarray(mesh.elem_size, dtype=float)))

    def draw_outline(self, name, selection, colour):
        """The selection's shape as edges, always in OUTLINE_COLOUR, apart from every colour the nodes use
        (place_outline); nothing for a place without one. colour is the nodes', kept for the call."""
        from lsr_mmto.problem import selections as S
        self.plotter.remove_actor(name, reset_camera=False)
        shape = place_outline(S.resolved(selection, self.project.selections), self.preview_mesh())
        if shape is not None:
            self.plotter.add_mesh(shape, color=OUTLINE_COLOUR, style="wireframe", line_width=2, pickable=False, name=name,
                                  reset_camera=False)       # edges, not a see-through surface: transparency
                                                            # can hide the text over the 3D view

    def draw_one(self, name, colour, bc_index=None):
        """Only one selection on the part (a row clicked in Supports and loads): its nodes, its shape, and the
        arrow of the load at bc_index. Returns its node count."""
        p, mesh = self.project, self.preview_mesh()
        self.clear_preview()
        xyz = np.asarray(mesh.node_xyz, dtype=float)
        nodes = p.evaluate(name, mesh, "nodes")
        self.draw_points(f"mmto_{name}", xyz[nodes], colour)
        self.draw_outline("mmto_outline", p.selections[name], colour)
        bc = p.bcs[bc_index] if bc_index is not None else None
        d = load_direction(bc) if bc is not None else None
        if d is not None and len(nodes) and np.linalg.norm(d) > 0:
            self._arrow(bc_index, d, xyz[nodes], float(np.linalg.norm(np.ptp(xyz, axis=0))))
        self.plotter.render()
        return len(nodes)

    def draw_supports_and_loads(self):
        """Every region on the run's mesh: supports blue, loads red with an arrow each, other regions
        orange, keep-solid elements grey. Returns (report lines, {region: node count})."""
        import pyvista as pv
        from lsr_mmto.problem.boundary_conditions import BC_KINDS
        p, mesh = self.project, self.preview_mesh()
        self.clear_preview()
        xyz, centres = np.asarray(mesh.node_xyz, dtype=float), np.asarray(mesh.elem_centers, dtype=float)
        role = {}
        for bc in p.bcs:
            if bc.get("kind") in BC_KINDS:
                role.setdefault(bc["on"], set()).add(BC_KINDS[bc["kind"]].role)
        lines, counts = [], {}
        for name, s in p.selections.items():
            nodes = p.evaluate(name, mesh, "nodes")
            counts[name] = len(nodes)
            colour = (SUPPORT_COLOUR if "support" in role.get(name, ()) else
                      LOAD_COLOUR if "load" in role.get(name, ()) else OTHER_COLOUR)
            if name in role or name not in p.keep_solid_names:
                self.draw_points(f"mmto_{name}", xyz[nodes], colour)
            if name in p.keep_solid_names:                         # the kept elements, as grey blocks
                elems = p.evaluate(name, mesh, "elements")
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
            d = load_direction(bc)
            if d is None or bc.get("on") not in p.selections:
                continue
            nodes = p.evaluate(bc["on"], mesh, "nodes")
            if np.linalg.norm(d) == 0 or len(nodes) == 0:
                continue
            self._arrow(i, d, xyz[nodes], size)
        try:
            bcs, _ = p.supports_and_loads(mesh)
            error = p.supports_and_loads_error(mesh, bcs) if p.bcs else None
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
        self.show_setup_view()
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
        self.plotter.remove_actor("geometry_info", reset_camera=False)
        if after and after != before:
            self.project.stl = os.path.abspath(after)
            if self.own_folder():                                   # the new part goes into the project folder
                from lsr_mmto.problem.project_folder import PART, copy_in
                self.project.stl = copy_in(self.project.stl, self.project.base_dir, PART)
            self._design = None                                     # a design of the old part no longer fits
            self.view = "setup"
            self._view_changed()
            self.draw_ruler()
            if names:
                self.message_text.append(f"Check the selections {', '.join(names)}: they are triangle numbers "
                                         f"of the old STL.")
            self._refresh_title()


def install_error_handler(window):
    """An error inside the app (in what a button or a field does) shown in a message box and written to the
    log, with the app left running. Without it, PyQt ends the whole app at once and unsaved work is lost."""
    import traceback

    def handler(kind, value, tb):
        if issubclass(kind, KeyboardInterrupt):
            return sys.__excepthook__(kind, value, tb)
        print("".join(traceback.format_exception(kind, value, tb)), file=sys.stderr, flush=True)
        try:
            window.message_text.append(f"Error: {kind.__name__}: {value}")
            QtWidgets.QMessageBox.critical(
                window, APP, f"Something went wrong:\n\n{kind.__name__}: {value}\n\nLSR MMTO is still "
                             f"running. If anything looks wrong, save the project and start the app again.")
        except Exception:                                           # noqa: BLE001
            pass
    sys.excepthook = handler
    return handler


def main(argv=(), splash=None):
    """The window, with argv[0] open if given; splash (run_gui.py's start-up picture) closes when it shows."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyleSheet("* { font-size: 10pt; }")                    # as PyTO-AD's main()
    from PyQt5.QtGui import QIcon
    app.setWindowIcon(QIcon(ICON))                                 # title bar and taskbar
    window = MMTOWindow()
    install_error_handler(window)
    window.show()
    if splash is not None:                                          # closed outright: finish() can wait forever
        app.processEvents()
        splash.close()
    if argv:
        window.open_project(argv[0])
    return app.exec_()
