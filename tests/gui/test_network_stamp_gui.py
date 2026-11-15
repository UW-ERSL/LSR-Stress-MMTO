"""
test_network_stamp_gui.py
=========================
The record a trained network keeps of its database (materials/network_stamp.py) in the window, off-screen:
a run on a network whose database changed only in colours or labels asks once (No: nothing starts; Yes: the
network is stamped and not asked about again); Materials says when the database has changed since its network
was trained, and when the next run will ask (for an included database: to put it back or copy it); an analysis,
which uses no network, is not asked; the editor keeps every digit of a value it is not asked to change, and sets
the network aside after a rename; Open in Excel does not open a database that comes with LSR MMTO. The questions are answered by the test (database_editor.ask), never shown.

    python -m pytest tests/gui/test_network_stamp_gui.py
"""
import os
import shutil
import sys
import tempfile

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform.startswith("win"):                         # off-screen Qt on Windows finds no fonts itself
    os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from PyQt5 import QtCore, QtWidgets  # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
QtCore.qInstallMessageHandler(lambda kind, context, message: None if "OpenGL" in message or
                              "propagateSizeHints" in message else sys.stderr.write(message + "\n"))

from gui import database_editor as DE, main_window as G, materials_panel  # noqa: E402

DATABASE = os.path.join(ROOT, "data", "databases", "6Materials_AF_3Attr")
Box = QtWidgets.QMessageBox


@pytest.fixture
def window():
    """The U-bracket open in a window from a copy in a scratch folder, its database the project's own copy of
    the 6-material database, saved as the editor saves it, with its network stamped with it."""
    from lsr_mmto.materials.network_stamp import stamp
    d = tempfile.mkdtemp(prefix="_test_gui_", dir=ROOT)
    try:
        w = G.MMTOWindow()
        assert w.open_project(shutil.copy(os.path.join(ROOT, "projects", "U-bracket.mmto"), d))
        os.makedirs(os.path.join(d, "materials"))
        xlsx = os.path.join(d, "materials", "steels.xlsx")
        shutil.copy(DATABASE + ".xlsx", xlsx)
        shutil.copy(DATABASE + ".nt", os.path.join(d, "materials", "steels.nt"))
        _rewritten(xlsx, lambda *a: None)
        stamp(os.path.join(d, "materials", "steels.nt"), xlsx)
        w.project.database = os.path.relpath(xlsx, w.project.base_dir)
        yield w, xlsx
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def answers(monkeypatch):
    """The questions asked, and the answers to give, in order: the test fills answers before an action."""
    asked, given = [], []

    def answer(parent, title, text, buttons, default, words=None):
        asked.append((title, text))
        return given.pop(0)
    monkeypatch.setattr(DE, "ask", answer)
    return asked, given


def _rewritten(xlsx, change):
    """The database written again as the editor saves it, after change(names, labels, colours, values)."""
    from lsr_mmto.materials.databases import database_table, material_display, write_database
    names, attributes, units, values = database_table(xlsx)
    colours, labels = material_display(xlsx)
    names, labels, colours, values = list(names), list(labels), list(colours), values.copy()
    change(names, labels, colours, values)
    write_database(xlsx, attributes, units, names, labels, colours, values.tolist())


def test_a_run_asks_once_about_a_network_whose_database_changed_only_in_colour(window, answers):
    """No: nothing starts, and the status says why. Yes: stamped, and not asked again. An analysis is not asked."""
    w, xlsx = window
    asked, given = answers
    run = w.open_run_panel()
    assert run._checked() and asked == []                               # nothing changed: nothing asked
    _rewritten(xlsx, lambda names, labels, colours, values: colours.__setitem__(0, "#123456"))
    assert run._checked(network=False) and asked == []                 # an analysis uses no network
    given.append(Box.No)
    assert not run._checked()
    assert asked[-1][0] == "Check the network" and "colour #123456, was #f58231" in asked[-1][1], asked
    assert run.status.text() == "Not started: the network of steels.xlsx was not confirmed."
    given.append(Box.Yes)
    assert run._checked() and len(asked) == 2
    assert run._checked() and len(asked) == 2                           # not asked again


def test_materials_says_what_the_network_needs(window, answers):
    """Nothing when the network matches; the question to come for a label changed; an error naming the cell for a
    number changed; for an included database, to put it back or make a copy."""
    from lsr_mmto.materials import network_stamp
    w, xlsx = window
    m = materials_panel.MaterialsPanel(w, w.project)
    assert not any(x.startswith(("Error: the database has changed", "The next run asks"))
                   for x in m.report.toPlainText().split("\n")), m.report.toPlainText()
    _rewritten(xlsx, lambda names, labels, colours, values: labels.__setitem__(0, "First"))
    m = materials_panel.MaterialsPanel(w, w.project)
    assert "The next run asks once about the network: only colours or labels changed since its network was " \
           "trained: " in m.report.toPlainText(), m.report.toPlainText()

    def double(names, labels, colours, values):
        values[0, 0] = 2 * values[0, 0]
    _rewritten(xlsx, double)
    m = materials_panel.MaterialsPanel(w, w.project)
    line = next(x for x in m.report.toPlainText().split("\n") if x.startswith("Error: the database has changed"))
    assert "cell B3" in line and "Versions..." in line, line
    assert any("the database has changed since its network was trained" in str(e) for e in w.project.validate())

    w.project.database = "6Materials_AF_3Attr"
    real = network_stamp.compare
    network_stamp.compare = lambda xlsx, nt: network_stamp.Comparison(network_stamp.DISPLAY, "only labels changed")
    try:
        m = materials_panel.MaterialsPanel(w, w.project)
        assert "Error: this database comes with LSR MMTO and does not match its network: only labels changed. Put " \
               "the file back as it came, or make your own copy with Edit..." in m.report.toPlainText()
    finally:
        network_stamp.compare = real


def test_the_editor_keeps_every_digit_and_sets_the_network_aside_after_a_rename(window):
    """A value with 16 significant digits shows and saves as it is: a label changed keeps the network (the
    next run asks once). A material renamed sets the network aside, as a number changed does."""
    from lsr_mmto.materials.databases import database_table
    from lsr_mmto.materials.network_stamp import DISPLAY, compare, stamp
    w, xlsx = window
    nt = xlsx[:-5] + ".nt"
    _rewritten(xlsx, lambda names, labels, colours, values: values.__setitem__((0, 0), 0.1234567890123456))
    stamp(nt, xlsx)
    e = DE.DatabaseEditor(None, os.path.dirname(xlsx), path=xlsx)
    assert e.table.item(1, 3).text() == "0.1234567890123456"
    e.table.item(1, 1).setText("First")
    e.save()
    assert e.network_set_aside is None and compare(xlsx, nt).state == DISPLAY
    assert database_table(xlsx)[3][0, 0] == 0.1234567890123456
    e = DE.DatabaseEditor(None, os.path.dirname(xlsx), path=xlsx)
    e.table.item(2, 0).setText("Renamed")
    e.save()
    assert e.network_set_aside == xlsx[:-5] + ".previous.nt" and not os.path.exists(nt)


def test_open_in_excel_leaves_an_included_database_alone(monkeypatch):
    """Nothing is opened; the report says to make a copy with Edit... first."""
    opened = []
    monkeypatch.setattr(materials_panel.subprocess, "Popen", lambda *a, **k: opened.append(a))
    if hasattr(materials_panel.os, "startfile"):
        monkeypatch.setattr(materials_panel.os, "startfile", lambda *a, **k: opened.append(a))
    d = tempfile.mkdtemp(prefix="_test_gui_", dir=ROOT)
    try:
        w = G.MMTOWindow()
        assert w.open_project(shutil.copy(os.path.join(ROOT, "projects", "U-bracket.mmto"), d))
        m = materials_panel.MaterialsPanel(w, w.project)
        m.excel_button.click()
        assert opened == [] and m.report.toPlainText().startswith(materials_panel.INCLUDED_IN_EXCEL)
    finally:
        shutil.rmtree(d, ignore_errors=True)
