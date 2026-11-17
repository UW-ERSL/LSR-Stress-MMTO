"""
test_gui_look.py
================
The LSR MMTO window's look and speed, off-screen: the part's facts come from one fast sum equal to
PyTO-AD's, and a large part opens and draws its ruler in a second or two; plots keep the size they
were made at, so the panels holding them do not creep wider.

    python tests/gui/test_gui_look.py        (or: python -m pytest tests)
"""
import glob
import os
import re
import sys
import tempfile
import time

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform.startswith("win"):                         # off-screen Qt on Windows finds no fonts itself
    os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from PyQt5 import QtCore, QtGui, QtTest, QtWidgets  # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _quiet(kind, context, message):
    if not any(s in message for s in ("OpenGL", "propagateSizeHints")):
        sys.stderr.write(message + "\n")


QtCore.qInstallMessageHandler(_quiet)

from gui import main_window as G  # noqa: E402

PROJECTS = os.path.join(ROOT, "projects")


def settle(seconds=0.0):
    end = time.time() + seconds
    APP.processEvents()
    while time.time() < end:
        APP.processEvents()


def window(project="L-bracket", size=(1225, 790)):
    w = G.MMTOWindow()
    w.resize(*size)
    w.show()
    w.open_project(os.path.join(PROJECTS, project + ".mmto"))
    settle()
    return w


def test_part_mass_properties_equal_pyto_ad():
    """The numpy sum gives PyTO-AD's area, volume, centre and inertia on every shipped part."""
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "models", "*", "*.[sS][tT][lL]"))):
        geom = G.STLGeom(path)
        ref = geom.compute_mass_properties()
        new = G.part_mass_properties(geom)
        assert np.isclose(new[0], ref[0], rtol=1e-10, atol=0), path
        assert np.isclose(new[1], ref[1], rtol=1e-10, atol=0), path
        assert np.allclose(new[2], ref[2], rtol=1e-10, atol=1e-12 * np.max(np.abs(ref[2]))), path
        assert np.allclose(new[3], ref[3], rtol=1e-10, atol=1e-12 * np.max(np.abs(ref[3]))), path
        assert G.part_mass_properties(geom) is new                  # kept, not summed again


def test_large_part_opens_fast():
    """A part of about 100 000 triangles: New project and the ruler each take a second or two (they took
    43 s and 20 s when the volume was summed triangle by triangle, three times over)."""
    import pyvista as pv
    with tempfile.TemporaryDirectory() as d:
        stl = os.path.join(d, "ball.stl")
        pv.Sphere(radius=50, theta_resolution=260, phi_resolution=200).triangulate().save(stl)
        w = G.MMTOWindow()
        w.show()
        t = time.time()
        assert w.new_project(stl=stl, folder=os.path.join(d, "ball"))
        settle()
        assert time.time() - t < 6.0
        t = time.time()
        w.ruler_box.setChecked(True)
        settle()
        assert time.time() - t < 2.0
        w.close()


def test_plot_keeps_its_preferred_size():
    """A plot's preferred size is the size it was made at, whatever size it is given later: the panels
    holding plots no longer creep wider."""
    from matplotlib.figure import Figure
    from gui.run_panel import plot_canvas
    canvas = plot_canvas(Figure(figsize=(4, 2.6)))
    first = canvas.sizeHint()
    canvas.resize(900, 700)
    canvas.show()
    settle()
    assert canvas.sizeHint() == first
    canvas.close()


def test_current_run_panel_keeps_its_width():
    """Opening Current run again and again leaves its width where it was."""
    w = window()
    w.open_run_panel()
    settle(0.3)
    dock = w.run_window.parentWidget()
    width = dock.width()
    for _ in range(10):
        w.open_run_panel()
        settle(0.1)
    assert abs(dock.width() - width) <= 2
    w.close()


def test_faces_picked_only_in_the_pick_strip():
    """A left click on the part selects faces only while the Pick strip is open, and writes no message
    line; the count is in the status bar. A right click clears them, again only with Pick open."""
    w = window()
    part = w.part_actor()

    class Picker:
        def GetCellId(self):
            return 3

        def GetActor(self):
            return part
    lines = w.message_text.toPlainText()
    w.on_left_button_press([0.0, 0.0, 1.0], Picker())              # turning the view: nothing selected
    w.on_right_button_press(None, None)
    assert w.picked_count() == 0 and w.message_text.toPlainText() == lines
    w.open_structural_loads_window()
    settle(0.2)
    w.on_left_button_press([0.0, 0.0, 1.0], Picker())
    n = w.picked_count()
    assert n > 0 and w.measured.text() == f"{n} triangle{'s' if n != 1 else ''} selected"
    assert w.message_text.toPlainText() == lines
    w.on_right_button_press(None, None)
    assert w.picked_count() == 0 and w.measured.text() == ""
    assert w.message_text.toPlainText() == lines
    w.close()


def test_summary_in_sections():
    """A finished run's Summary: the run, then Result (the objective, each constraint met or not), Design,
    Materials and Run, as tables; the comparison of two runs in the same sections."""
    from lsr_mmto.problem.project import Project, run
    from lsr_mmto.results import compare, tables
    from lsr_mmto.results.load_results import load
    from gui.run_panel import summary_view
    with tempfile.TemporaryDirectory() as d:
        p = Project.load(os.path.join(PROJECTS, "L-bracket.mmto"))
        p.n_dof = 3000
        p.settings["common"]["maxIterations"] = 2
        run(p, save_dir=d, verbose=False)
        r = load(os.path.join(d, "run_data"))
    text = str(tables.summary(r))
    for section in ("Result", "Design", "Materials", "Run"):
        assert f"\n{section}\n" in text
    rows = [line for line in text.splitlines() if "volume fraction" in line]
    assert len(rows) == len(r.constraint_spec)
    assert all(line.rstrip().endswith(("met", "NOT MET")) for line in rows)
    assert "fraction of budget" not in text and "Mat1_Red volume fraction" in text
    view = summary_view(tables.summary(r), tables.summary_details(r))
    shown = view.findChild(QtWidgets.QTextBrowser).toPlainText()
    assert shown.count("\u2713 met") + shown.count("\u2717 not met") == len(r.constraint_spec)
    assert "Grey (in between)" in shown
    both = str(compare.table({"one": r, "two": r}))
    for section in ("Result", "Design", "Materials", "Run"):
        assert f"\n{section}\n" in both
    assert both.count(" met") + both.count("NOT MET") == 2 * len(r.constraint_spec)


def test_display_options():
    """View (PyTO-AD's Display Options) in plain words, without the controls that have nothing to act on, and its cuts
    work: a cut along x draws the part clipped, and back to 0 draws it whole again."""
    w = window()
    w.open_display_options_window()
    d = w._display_options_dialog
    settle(0.2)
    shown = [x.text() for x in d.findChildren((QtWidgets.QLabel, QtWidgets.QCheckBox, QtWidgets.QPushButton))
             if x.isVisible()]
    assert shown == ["Cut along x (%)", "Cut along y (%)", "Cut along z (%)",
                     "Show the part's triangles", "See-through part", "Save Image", "Close"]
    assert d.windowTitle() == "View" and w.view_button.isEnabled() and "Display Options" not in G.SIDEBAR
    part = w.part_actor()
    whole = part.GetMapper().GetInput().GetNumberOfCells()
    d.x_cutting_spin.setValue(40)
    settle()
    cut = [a for n, a in w.plotter.actors.items() if str(n).startswith("cut_")]
    assert len(cut) == 1 and not part.GetVisibility()
    assert 0 < cut[0].GetMapper().GetInput().GetNumberOfCells() <= 2 * whole
    d.x_cutting_spin.setValue(0)
    settle()
    assert part.GetVisibility() and not any(str(n).startswith("cut_") for n in w.plotter.actors)
    assert "geometry_info" not in w.plotter.actors                    # no text over the 3D view
    w.close()


def test_view_leaves_the_scene_as_it_was():
    """Every View option ticked and unticked, twice, and every cut set and set back: the 3D view holds the same
    drawings as before, each visible as before and over the same space. (Ticking the bounding box once left
    a box from -1 to 1 on every axis behind for good; the box is no longer offered.)"""
    def scene():
        out = []
        for a in w.plotter.actors.values():
            visible = a.GetVisibility() if hasattr(a, "GetVisibility") else None
            bounds = tuple(round(x, 6) for x in a.GetBounds()) if hasattr(a, "GetBounds") else None
            out.append((visible, bounds))
        return sorted(out, key=str)
    w = window("U-bracket")
    before = scene()
    w.open_display_options_window()
    d = w._display_options_dialog
    settle(0.2)
    boxes = [b for b in d.findChildren(QtWidgets.QCheckBox) if b.isVisible()]
    assert len(boxes) == 2
    for _ in range(2):
        for box in boxes:
            box.setChecked(not box.isChecked())
            settle()
            box.setChecked(not box.isChecked())
            settle()
    for spin in (d.x_cutting_spin, d.y_cutting_spin, d.z_cutting_spin):
        spin.setValue(50)
        settle()
        spin.setValue(0)
        settle()
    assert scene() == before, (before, scene())
    assert not any("BoundingBox" in str(n) or "bounding_box" in str(n) for n in w.plotter.actors)
    w.close()


def test_no_window_has_the_help_button():
    """Windows' "?" help button, which has no help to show, is off for every window once the app's style is on
    (main_window.apply_style), the windows inherited from PyTO-AD too; and no window turns it off on its own,
    so there is one way of doing it."""
    G.apply_style(APP)
    try:
        for window in (QtWidgets.QDialog(), QtWidgets.QMessageBox(), G.DisplayOptions(G.MMTOWindow())):
            assert not window.windowFlags() & QtCore.Qt.WindowContextHelpButtonHint, window
    finally:
        APP.setStyleSheet("")
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_DisableWindowContextHelpButton, False)
    for path in glob.glob(os.path.join(ROOT, "gui", "*.py")):
        with open(path, encoding="utf-8") as f:
            assert "WindowContextHelpButtonHint" not in f.read(), path


def test_a_control_that_is_off_looks_off():
    """With the app's style, every kind of control the panels use looks different when it is off (greyed) from
    when it is on, a ticked checkbox and a chosen round button too (they once looked the same, blue, with dark
    text): the style greys any control that is off, whatever its kind. The Materials table and the database
    editor draw every cell and header centred, cells made later too."""
    from gui import database_editor, materials_panel, panel_widgets
    makers = {"checkbox": lambda: QtWidgets.QCheckBox("Distance penalty"),
              "round button": lambda: QtWidgets.QRadioButton("Single material"),
              "text box": lambda: QtWidgets.QLineEdit("0.5"), "number box": lambda: QtWidgets.QSpinBox(),
              "list": lambda: QtWidgets.QComboBox(), "button": lambda: QtWidgets.QPushButton("Reset"),
              "label": lambda: QtWidgets.QLabel("Weight"), "tool button": lambda: QtWidgets.QToolButton()}
    G.apply_style(APP)
    try:
        for kind, make in makers.items():
            pictures = []
            for on in (True, False):
                widget = make()
                if isinstance(widget, (QtWidgets.QCheckBox, QtWidgets.QRadioButton)):
                    widget.setChecked(True)
                if isinstance(widget, QtWidgets.QComboBox):
                    widget.addItem("ORIGIN")
                if isinstance(widget, QtWidgets.QToolButton):
                    widget.setText("Reset")
                widget.setEnabled(on)
                widget.resize(160, 30)
                widget.show()
                settle(0.05)
                pictures.append(widget.grab().toImage())
                widget.close()
            assert pictures[0] != pictures[1], f"a {kind} looks the same off as on"
        for make in (lambda: QtWidgets.QCheckBox("Distance penalty"), lambda: QtWidgets.QRadioButton("One")):
            boxes = []                                  # the tick box or round button alone, without its text
            for on in (True, False):
                widget = make()
                widget.setChecked(True)
                widget.setEnabled(on)
                widget.resize(160, 30)
                widget.show()
                settle(0.05)
                option = QtWidgets.QStyleOptionButton()
                widget.initStyleOption(option)
                box = widget.style().subElementRect(QtWidgets.QStyle.SE_CheckBoxIndicator if
                                                    isinstance(widget, QtWidgets.QCheckBox) else
                                                    QtWidgets.QStyle.SE_RadioButtonIndicator, option, widget)
                boxes.append(widget.grab(box).toImage())
                widget.close()
            assert boxes[0] != boxes[1], f"a ticked {type(widget).__name__}'s box looks the same off as on"
    finally:
        APP.setStyleSheet("")
    w = window()
    m = materials_panel.MaterialsPanel(w, w.project)
    e = database_editor.DatabaseEditor(None, w.project.base_dir)
    for view in (m.table, e.table, e.table.frozen):
        assert isinstance(view.itemDelegate(), panel_widgets._Centred), view
        assert view.horizontalHeader().defaultAlignment() == QtCore.Qt.AlignCenter
    index = e.table.model().index(0, 1)                 # a value cell: typed in the middle too
    editor = e.table.itemDelegate().createEditor(e.table.viewport(), QtWidgets.QStyleOptionViewItem(), index)
    assert isinstance(editor, QtWidgets.QLineEdit) and editor.alignment() & QtCore.Qt.AlignHCenter, editor
    e.done(0)


def test_every_look_has_a_greyed_look():
    """Every rule of the window's style sheet that colours text or a border has a greyed version for a control
    that is off, in the same states: written by hand, or added by panel_widgets.with_off_look (rules added later
    too). A widget's own sheet gets the same (panel_widgets.set_style), a bare "color: ..." too."""
    from gui import panel_widgets as P
    with open(G.STYLE, encoding="utf-8") as fh:
        sheet = P.with_off_look(fh.read())
    rules = P._rules(sheet)
    off = {P._look(sel) for selectors, _ in rules for sel in selectors if "disabled" in re.findall(P._STATE, sel)}
    for selectors, body in rules:
        if re.search(P._TEXT, body) or re.search(P._BORDER, body):
            for sel in selectors:
                what, states = P._look(sel)
                assert "disabled" in re.findall(P._STATE, sel) or (what, states) in off or \
                    states & P._POINTER and any(w == what for w, _ in off), f"{sel} has no greyed look"
    assert "QLabel#x:disabled { color: %s; border-color: %s; }" % (P.OFF_TEXT, P.OFF_BORDER) in \
        P.with_off_look("QLabel#x { color: red; border: 1px solid #ff0000; }")
    assert "hover:disabled" not in P.with_off_look("QCheckBox::indicator:disabled { border-color: #dfe3e8; }\n"
                                                   "QCheckBox::indicator:hover { border-color: #3b5bdb; }")
    assert P.with_off_look("color: #b00000;").endswith("*:disabled { color: %s; }" % P.OFF_TEXT)
    assert P.with_off_look("background: #ffd6d6;") == "background: #ffd6d6;"


def test_every_style_sheet_gets_its_greyed_look():
    """The window's code sets a style sheet only through panel_widgets.set_style (a widget's) and
    main_window.apply_style (the app's), which both add the greyed looks: a widget given its own look later gets
    its greyed look too."""
    import ast
    allowed = {("panel_widgets.py", "set_style"), ("main_window.py", "apply_style")}
    for path in glob.glob(os.path.join(ROOT, "gui", "**", "*.py"), recursive=True):
        tree = ast.parse(open(path, encoding="utf-8").read())
        inside = {}                                     # each node: the function it is written in
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for node in ast.walk(fn):
                    inside[node] = fn.name              # walked outer first, so the innermost name stays
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "setStyleSheet":
                assert (os.path.basename(path), inside.get(node)) in allowed, \
                    f"{os.path.relpath(path, ROOT)}:{node.lineno} sets a style sheet itself: use set_style"


def test_stop_and_the_info_button_look_off_when_off():
    """Stop, once pressed, is off and grey, no longer red; an "i" button that is off is grey too. Both have their
    own look, which the general greyed look alone did not reach."""
    from gui import panel_widgets as P
    styled()
    try:
        row = QtWidgets.QWidget()                       # the top row as the window names it (its timer keeps
        row.setObjectName("topButtons")                 # turning the window's own button back on)
        b = QtWidgets.QToolButton(row)
        b.setObjectName("stopButton")
        b.setText("Stop")
        row.show()
        for on in (True, False):
            b.setEnabled(on)
            settle(0.05)
            image = b.grab().toImage()
            red = sum(1 for x in range(image.width()) for y in range(image.height())
                      if (lambda c: c.red() > 150 and c.green() < 110 and c.blue() < 110)(image.pixelColor(x, y)))
            assert (red > 0) == on, f"Stop {'on' if on else 'off'}: {red} red pixels"
        pictures = []
        for on in (True, False):
            info = P.Info("text")
            info.setEnabled(on)
            info.show()
            settle(0.05)
            pictures.append(info.grab().toImage())
            info.close()
        assert pictures[0] != pictures[1], "an i button looks the same off as on"
        row.close()
    finally:
        APP.setStyleSheet("")


def styled():
    """The window's style sheet on the app, as main() puts it; cleared again by the caller."""
    G.apply_style(APP)
    settle()


def test_top_row_and_bottom_buttons():
    """One top row: LSR MMTO, File and Help (their menus and shortcuts), Part / Design, the light / dark
    switch, View, Export STL, Run, Analyse; in a window held at 1225 px the row fits and nothing in it is cut
    short; while a run goes (View and Analyse step aside), a progress text longer than the room left ends in an
    ellipsis (a shorter one shows whole).
    Snap, Measure and Ruler at the bottom drive the window's measuring and scales."""
    styled()
    try:
        w = window()
        w.setFixedWidth(1225)
        assert not w.menuBar().isVisible()
        menus = {b.text(): [a.text() for a in b.menu().actions()] for b in w.findChildren(QtWidgets.QToolButton)
                 if b.objectName() == "menuButton"}
        assert set(menus) == {"&File", "&Help"} and "&Save" in menus["&File"]
        assert {a.shortcut().toString() for a in w.actions()} >= {"Ctrl+N", "Ctrl+O", "Ctrl+S"}
        assert [b.text() for b in (w.part_button, w.design_button, w.view_button, w.export_button, w.run_button,
                                   w.analyse_button)] == ["Part", "Design", "View", "Export STL", "Run", "Analyse"]
        assert not w.design_button.isEnabled() and w.part_button.isChecked()      # no design yet
        w.update_top_row()
        assert not w.export_button.isEnabled() and w.shown_run() is None          # nothing to export
        p = w.progress_text
        w._top_timer.stop()                                   # it would clear the text: no run goes
        for text, whole, idle in (("", True, True),            # nothing going: Analyse beside Run
                                  (" ".join(["Starting: building the mesh and the material network..."] * 3),
                                   False, False),                   # longer than the room in any font
                                  ("Iteration 6 / 40 \u00b7 0:09 so far \u00b7 about 0:46 left", True, False)):
            w.analyse_button.setVisible(idle)                   # as update_top_row sets them while a run goes
            w.view_button.setVisible(idle)
            p.setText(text)
            row = w.top_row.widgetForAction(w.top_row.actions()[0])
            room = w.top_row.layout().contentsRect().width()
            end = time.time() + 5                               # the tool bar takes its new layout (animated)
            while row.width() != min(row.sizeHint().width(), room) and time.time() < end:
                APP.processEvents()
            assert w.width() == 1225 and row.isVisible() and row.width() >= row.minimumSizeHint().width()
            assert w.top_row.contentsRect().contains(row.geometry())
            cut = [x for x in row.findChildren((QtWidgets.QLabel, QtWidgets.QToolButton))
                   if x.isVisible() and x is not p and x.width() < x.sizeHint().width()]
            assert not cut, [(x.objectName(), x.text()) for x in cut]
            shown = p.fontMetrics().elidedText(text, QtCore.Qt.ElideRight, p.contentsRect().width())
            assert (shown == text) == whole and (whole or shown.startswith("Starting: building")), shown
        w.ruler_box.click()
        assert w.ruler_box.isChecked() and w._rulers
        w.measure.click()
        assert w.measured.text() == "Measure: click the first point"
        w.measure.setChecked(False)                                       # (Esc)
        assert w.measured.text() == ""
        w.snap.setCurrentIndex(2)
        assert w.snapped(np.array([0.4, 1.6, 0.0])).tolist() == [0.0, 2.0, 0.0]
        w.close()
    finally:
        APP.setStyleSheet("")


def test_every_tab_text_fits_its_tab():
    """The tab shown is marked by a line along its top, not bold text, so Qt's tab sizes (worked out with
    the plain font) hold its text: in the Settings window and Results (and the strips' own tabs) every
    tab's text and padding fit inside its tab, whichever tab is shown."""
    from gui import settings_panel
    styled()
    try:
        w = window()
        settings = settings_panel.SettingsPanel(w, w.project)
        settings.show()
        w.open_supports_and_loads_panel()
        w.open_run_panel()
        res = w.open_results_panel()
        for title in ("Convergence", "Latent space", "Summary"):           # the tabs a run shows
            res.tabs.addTab(QtWidgets.QLabel(title), title)
        settle(0.2)
        bars = settings.findChildren(QtWidgets.QTabBar) + w.findChildren(QtWidgets.QTabBar)
        assert any(b.count() >= 2 for b in settings.findChildren(QtWidgets.QTabBar))
        assert res.tabs.tabBar() in bars
        with open(G.STYLE, encoding="utf-8") as f:
            selected = re.search(r"QTabBar::tab:selected\s*\{([^}]*)\}", f.read()).group(1)
        assert "border-top: 2px solid #2f5fe0" in selected                  # the mark of the tab shown
        checked = 0
        for bar in bars:
            if not bar.isVisible():
                continue
            shown = QtGui.QFont(bar.font())                                # the font the tab shown is drawn in
            shown.setBold("font-weight" in selected and "normal" not in selected)
            for current in range(bar.count()):
                bar.setCurrentIndex(current)
                settle()
                for i in range(bar.count()):
                    text = bar.tabText(i).replace("&", "")
                    font = QtGui.QFontMetrics(shown) if i == current else bar.fontMetrics()
                    need = font.horizontalAdvance(text) + 2 * 14 + 2 * 1 + 2       # padding, borders, margin
                    assert need <= bar.tabRect(i).width(), (text, need, bar.tabRect(i).width())
                    checked += 1
        assert checked > 20
        settings.close()
        w.close()
    finally:
        APP.setStyleSheet("")


def test_light_by_default_and_dark_on_the_switch():
    """The 3D view starts light. In the dark view the window's own drawings are drawn light from the
    first frame (the Measure line here), and back in the light view they are dark again; the part keeps
    its colour."""
    from gui import dark_and_light_3d_view as view_mode
    w = window()
    assert not w.dark_view and tuple(w.plotter.renderer.GetBackground()) == (1.0, 1.0, 1.0)
    part_colour = w.part_actor().GetProperty().GetColor()
    w.theme_button.click()
    assert w.dark_view
    w._measure_points = [np.zeros(3), np.array([10.0, 0.0, 0.0])]
    w._draw_measure()
    line = w.plotter.actors["measure_line"].GetProperty().GetColor()
    assert min(line) > 0.7                                                # light on the dark view
    view_mode.apply(w.plotter, w.dark_view)                               # the next frame leaves it alone
    assert w.plotter.actors["measure_line"].GetProperty().GetColor() == line
    assert w.part_actor().GetProperty().GetColor() == part_colour
    w.theme_button.click()
    assert not w.dark_view
    assert max(w.plotter.actors["measure_line"].GetProperty().GetColor()) < 0.2
    w.close()


def test_measure_white_in_the_dark_view():
    """Measure's line and points are bright white in the dark view and #222222 in the light view, drawn in
    either view and after a switch both ways; another dark drawing of the same colour is only lightened."""
    import pyvista as pv
    from gui import dark_and_light_3d_view as view_mode
    w = window()
    w._measure_points = [np.zeros(3), np.array([10.0, 0.0, 0.0])]
    dark_grey = view_mode._rgb("#222222")

    def colours():
        return [tuple(round(c, 6) for c in w.plotter.actors[n].GetProperty().GetColor())
                for n in ("measure_line", "measure_points")]
    w._draw_measure()
    assert colours() == [tuple(round(c, 6) for c in dark_grey)] * 2       # light view
    w.theme_button.click()
    assert colours() == [(1.0, 1.0, 1.0)] * 2                             # switched to dark
    other = w.plotter.add_mesh(pv.Line((0, 0, 0), (0, 5, 0)), color="#222222", name="plain_dark_line")
    w.plotter.render()
    view_mode.apply(w.plotter, True)
    assert 0.7 < max(other.GetProperty().GetColor()) < 1.0                # not white: no colour matching
    w._draw_measure()
    assert colours() == [(1.0, 1.0, 1.0)] * 2                             # drawn in the dark view
    w.theme_button.click()
    assert colours() == [tuple(round(c, 6) for c in dark_grey)] * 2       # and back
    w.close()


def test_message_panel_merges_repeats():
    """A line repeated at once is shown once, with how many times."""
    w = window()
    w.message_text.clear()
    for _ in range(3):
        w.message_text.append("Saved.")
    w.message_text.append("Done.")
    assert w.message_text.toPlainText().splitlines() == ["Saved.  (\u00d73)", "Done."]
    assert "-0 " not in w.extent.text() and "-0." not in w.extent.text()
    w.close()


def test_message_panel_problems_then_log():
    """The message panel: the project's errors (red) and warnings (amber) as they are now at the top, a
    thin line, then what happened. A problem fixed disappears, and shows again when it comes back; with no
    problems only the log is there. Clear empties the log and keeps the problems; the panel keeps its
    height."""
    w = window()
    height = w.message_frame.height()
    young = "Warning: the database gives Young's modulus"
    assert w.problems.isVisible() and w.problems_line.isVisible()
    assert young in w.problems.toPlainText() and young not in w.message_text.toPlainText()
    assert "#9a5b00" in w.problems.toHtml()                               # amber
    w.project.units = "MKS"                                               # fixed: the database is in MKS
    w._refresh_title()
    settle()
    assert not w.problems.isVisible() and not w.problems_line.isVisible() and w.message_text.isVisible()
    w.project.units = "no such units"                                     # an error
    w._refresh_title()
    assert w.problems.toPlainText().startswith("Error: units") and "#b00000" in w.problems.toHtml()
    w.project.units = "mmKS"                                              # broken again: shown again
    w._refresh_title()
    settle()
    assert w.problems.isVisible() and w.problems.toPlainText().count(young) == 1
    assert "Error" not in w.problems.toPlainText()
    w.message_text.append("Saved.")
    assert "Saved." in w.message_text.toPlainText()
    assert w.clear_log.toolTip() and not w.clear_log.icon().isNull()
    w.clear_log.click()
    assert w.message_text.toPlainText() == "" and young in w.problems.toPlainText()
    assert w.problems.height() < w.message_text.height() + w.problems.height() <= w.message_frame.height()
    assert w.message_frame.height() == height
    w.close()


def test_message_panel_resizable():
    """The divider between the 3D view and the message panel drags: the panel and its log get taller, the
    problems take at most half of the panel and the log the rest (all of it with no problems); a dragged
    height comes back in a new window (three lines at first); the start page, the legend and the banner stay
    in their places on the 3D view as it changes size."""
    G.view_settings().remove("message panel height")
    try:
        w = G.MMTOWindow()
        w.resize(1366, 860)
        w.show()
        settle()
        line = QtGui.QFontMetrics(w.message_text.font()).lineSpacing()
        assert w.message_frame.height() == G.MESSAGE_LINES * line + 48                # as it starts
        view, log = w.pv_frame.height(), w.message_text.height()
        w.splitter.setSizes([sum(w.splitter.sizes()) - 300, 300])                   # dragged taller
        settle()
        assert w.message_frame.height() == 300 and w.message_text.height() > log + 150
        assert w.pv_frame.height() < view - 150
        assert w.start_page.geometry() == w.pv_frame.rect().adjusted(1, 1, -1, -1)  # the start page follows
        w.open_project(os.path.join(PROJECTS, "L-bracket.mmto"))
        w.view, w.design_legend = "design", [[("#ff0000", "Red", 0.6), ("#0000ff", "Blue", 0.4)]]
        w.show_legend()
        w.banner.say("Exporting STL...")
        settle()
        panel = w.problems.panel
        w.problems.show_lines([("Warning", f"warning {k}") for k in range(30)])     # many problems: half
        settle()
        line = w.problems.fontMetrics().lineSpacing()
        assert panel.height() // 2 - line < w.problems.height() <= panel.height() // 2       # in whole lines
        assert w.message_text.height() >= panel.height() - w.problems.height() - 10
        w.problems.show_lines([])                                                  # none: the log has it all
        w.problems_line.hide()
        settle()
        assert w.message_text.height() == panel.height()
        w.splitter.setSizes([sum(w.splitter.sizes()) - 160, 160])
        settle()
        assert w.legend.geometry().topLeft() == QtCore.QPoint(12, 12) and w.legend.isVisible()
        assert w.banner.y() == 14 and w.banner.x() == max((w.pv_frame.width() - w.banner.width()) // 2, 8)
        sizes = w.splitter.sizes()
        w.splitter.moveSplitter(sizes[0] - 90, 1)                                   # a drag by the mouse
        kept = w.splitter.sizes()[1]
        assert kept == 250 and G.view_settings().value("message panel height", type=int) == kept
        w.view = "setup"
        w.close()
        w2 = G.MMTOWindow()
        w2.resize(1366, 860)
        w2.show()
        settle()
        assert w2.message_frame.height() == kept                                    # the next session
        w2.close()
    finally:
        G.view_settings().remove("message panel height")


def test_run_layout():
    """While a run goes the sidebar steps aside and Current run takes about half the window; afterwards
    both are back as they were."""
    styled()
    try:
        w = window()
        rw = w.open_run_panel()
        settle(0.3)
        dock = rw.parentWidget()
        before = dock.width()
        assert w.sidebar.isVisible()
        rw.optimization_running = True                                    # as while a run goes
        w.update_top_row()
        settle(0.3)
        assert not w.sidebar.isVisible() and abs(dock.width() - w.width() // 2) < 0.06 * w.width()
        assert w.run_button.text() == "Stop"
        rw.optimization_running = False
        w.update_top_row()
        settle(0.3)
        assert w.sidebar.isVisible() and abs(dock.width() - before) <= 4 and w.run_button.text() == "Run"
        w.close()
    finally:
        APP.setStyleSheet("")



def test_info_card():
    """An (i) click opens a card with its whole text in Markdown (paragraphs, bold, numbered steps), 420 px
    wide and wrapped, under the (i) and moved to stay inside the window at each edge, scrolling inside when
    taller than 60 % of the window; Esc closes it. The mouse resting on it shows only the first sentence."""
    from gui.panel_widgets import CARD_WIDTH, Info
    text = ("Import brings in a table of materials made elsewhere, as an Excel or CSV file. More here.\n\n"
            "**The file:** names in the first column, e.g. Density [kg/m^3].\n\n"
            "1. One step.\n2. Another step, with lbf*s^2/in in it.")
    top = QtWidgets.QWidget()
    top.resize(900, 600)
    grid = QtWidgets.QGridLayout(top)
    corners = {}
    for r, c in ((0, 0), (0, 2), (2, 0), (2, 2), (1, 1)):
        corners[(r, c)] = Info(text)
        grid.addWidget(corners[(r, c)], r, c, alignment=QtCore.Qt.AlignCenter)
    grid.setRowStretch(1, 1)
    grid.setColumnStretch(1, 1)
    top.show()
    settle(0.2)
    area = top.geometry()
    for where, info in corners.items():
        card = info.open_card()
        settle(0.05)
        box = card.frameGeometry()
        assert area.contains(box), (where, box, area)
        assert card.width() <= CARD_WIDTH + 30 and card.label.width() == CARD_WIDTH
        near = info.mapToGlobal(info.rect().center())
        assert abs(box.center().y() - near.y()) < box.height() / 2 + 40, where     # beside its (i)
        card.close()
        settle(0.05)
    card = corners[(1, 1)].open_card()
    settle(0.05)
    doc = card.label.text()
    assert "**The file:**" in doc and "\\*" in doc and "1. One step." in doc
    assert card.label.heightForWidth(CARD_WIDTH) > 4 * card.fontMetrics().height()            # wrapped
    QtTest.QTest.keyClick(card, QtCore.Qt.Key_Escape)
    settle(0.05)
    assert not card.isVisible()
    assert corners[(0, 0)].hover_text() == ("Import brings in a table of materials made elsewhere, as an Excel "
                                            "or CSV file.")
    long = Info("A first sentence. " + "Many words follow in this card. " * 200)
    grid.addWidget(long, 1, 0)
    settle(0.05)
    card = long.open_card()
    settle(0.05)
    assert card.height() <= 0.6 * top.geometry().height() + 4 and card.scroll.verticalScrollBar().maximum() > 0
    card.close()
    top.close()

if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(name, "ok")
