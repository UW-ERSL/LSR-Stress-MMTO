"""
test_gui_look.py
================
The LSR MMTO window's look and speed, off-screen: the part's facts come from one fast sum equal to
PyTO-AD's, and a large part opens and draws its ruler in a second or two; plots keep the size they
were made at, so the panels holding them do not creep wider.

    python tests/test_gui_look.py        (or: python -m pytest tests)
"""
import glob
import os
import sys
import tempfile
import time

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform.startswith("win"):                         # off-screen Qt on Windows finds no fonts itself
    os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PyQt5 import QtCore, QtWidgets  # noqa: E402

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


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(name, "ok")
