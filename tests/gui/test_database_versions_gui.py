"""
test_database_versions_gui.py
=============================
Older versions of a database in the window, off-screen: the editor's (modified) title and its two questions
(keeping the older version on Save, changes not saved on closing), the first version kept for a database
made, imported or copied from an included one, and Versions... in Materials with Restore and Delete. The
questions are answered by the test (database_editor.ask), never shown.

    python -m pytest tests/gui/test_database_versions_gui.py
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

from gui import database_editor as DE  # noqa: E402
from lsr_mmto.materials import database_versions as V  # noqa: E402

DATABASE = os.path.join(ROOT, "data", "databases", "6Materials_AF_3Attr")
Box = QtWidgets.QMessageBox


@pytest.fixture
def folder():
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "materials"))
    yield os.path.join(d, "materials")
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


def _own(folder):
    xlsx = os.path.join(folder, "steels.xlsx")
    shutil.copy(DATABASE + ".xlsx", xlsx)
    shutil.copy(DATABASE + ".nt", os.path.join(folder, "steels.nt"))
    return xlsx


def _value(xlsx):
    from lsr_mmto.materials.databases import database_table
    return database_table(xlsx)[3][0, 0]


def test_changes_are_marked_and_closing_with_them_asks(folder, answers):
    """Opened: no (modified). A value changed: (modified) in the title. Closing then asks: Cancel keeps the
    editor open, Discard changes closes it with the file as it was; Save... saves."""
    asked, given = answers
    xlsx = _own(folder)
    e = DE.DatabaseEditor(None, folder, path=xlsx)
    assert e.windowTitle() == "Database" and not e.modified()
    e.table.item(1, 3).setText("7")
    assert e.windowTitle() == "Database (modified)" and e.modified()
    given.append(Box.Cancel)
    e.reject()
    assert asked[-1] == ("Close the editor", "steels has changes that aren't saved.") and e.result() == 0
    given.append(Box.Discard)
    e.reject()
    assert e.result() == QtWidgets.QDialog.Rejected and _value(xlsx) == 1
    e = DE.DatabaseEditor(None, folder, path=xlsx)
    e.table.item(1, 3).setText("7")
    given.append(Box.Save)                                     # Save...: as Save, its first version kept
    e.reject()
    assert e.result() == QtWidgets.QDialog.Accepted and _value(xlsx) == 7 and V.has_first(xlsx)
    e = DE.DatabaseEditor(None, folder, path=xlsx)
    e.reject()                                                 # nothing changed: nothing asked
    assert len(asked) == 3 and e.result() == QtWidgets.QDialog.Rejected


def test_save_keeps_the_first_version_then_asks(folder, answers):
    """A database with no versions: its first Save keeps it as it was as the first version, without asking.
    Later Saves ask "Keep the older version?": Yes keeps it, No does not, Cancel does not save. A Save with
    nothing changed asks nothing."""
    asked, given = answers
    xlsx = _own(folder)
    for value, answer in ((7, None), (8, Box.Yes), (9, Box.No), (10, Box.Cancel)):
        e = DE.DatabaseEditor(None, folder, path=xlsx)
        e.table.item(1, 3).setText(str(value))
        if answer is not None:
            given.append(answer)
        e.accept()
    assert asked == [("Save steels", "Keep the older version?")] * 3
    kept = V.versions(xlsx)
    assert [v.first for v in kept] == [False, True] and [_value(v.path) for v in kept] == [7, 1]
    assert _value(xlsx) == 9                                    # Cancel saved nothing
    e = DE.DatabaseEditor(None, folder, path=xlsx)
    e.accept()
    assert len(asked) == 3 and len(V.versions(xlsx)) == 2


def test_a_new_database_and_a_copy_of_an_included_one_get_their_first_version(folder, answers):
    """Saved as new (imported here), its first version is itself; an included database edited, saved as a copy
    in the folder: the copy's first version is the included database as it came."""
    asked, given = answers
    imported = {"stem": "mine", "names": ["a", "b"], "attributes": ["Density", "Youngs_Modulus"],
                "units": ["kg/m^3", "Pa"], "rows": [["1", "100"], ["2", "200"]]}
    e = DE.DatabaseEditor(None, folder, imported=imported)
    e.accept()
    path = os.path.join(folder, "mine.xlsx")
    assert e.result() == QtWidgets.QDialog.Accepted and [v.first for v in V.versions(path)] == [True]
    e = DE.DatabaseEditor(None, folder, path=DATABASE + ".xlsx")
    e.table.item(1, 3).setText("7")
    e.accept()
    copy = e.saved_path
    assert os.path.dirname(copy) == folder and _value(copy) == 7
    first, = V.versions(copy)
    assert first.first and _value(first.path) == 1 and os.path.isfile(first.network)
    assert asked == []


def test_versions_restores_and_deletes(folder, answers):
    """Versions...: the versions, newest first. Restore asks "Keep the current version?" (Yes keeps it) and puts
    the chosen one back; Delete asks first, in words that say when it is the first version, and No keeps it."""
    from gui.database_versions_window import VersionsWindow
    asked, given = answers
    xlsx = _own(folder)
    V.keep(xlsx, first=True)
    e = DE.DatabaseEditor(None, folder, path=xlsx)
    e.table.item(1, 3).setText("7")
    given.append(Box.No)
    e.accept()
    w = VersionsWindow(None, xlsx)
    assert w.list.count() == 1 and "first version" in w.list.item(0).text() and "6 materials" in w.list.item(0).text()
    given.append(Box.Yes)
    assert w.restore() and _value(xlsx) == 1 and w.list.count() == 2   # the current (7) kept first
    assert asked[-1] == ("Restore steels", "Keep the current version?")
    w.list.setCurrentRow(1)                                    # the first version
    given.append(Box.No)
    assert not w.delete() and w.list.count() == 2
    assert asked[-1][1] == "This is the first version of steels. Delete it? This can't be undone."
    given.append(Box.Yes)
    assert w.delete() and w.list.count() == 1 and not V.has_first(xlsx)
    w.list.setCurrentRow(0)
    given.append(Box.Yes)
    assert w.delete() and w.list.count() == 0 and w.empty.isVisibleTo(w)


def test_a_save_that_changes_the_file_keeps_its_first_version(folder, answers):
    """A database with a column LSR MMTO does not know: the editor shows (modified) from the start, since Save
    writes it without that column, and that Save keeps the file as it was as the first version. Save... from
    the close question, then Cancel on "Keep the older version?": nothing saved, the editor still open."""
    from openpyxl import load_workbook
    asked, given = answers
    xlsx = _own(folder)
    book = load_workbook(xlsx)
    sheet = book.worksheets[0]
    column = sheet.max_column + 1
    for row, value in enumerate(["E0", "Pa"] + [1.0] * (sheet.max_row - 2), 1):
        sheet.cell(row=row, column=column).value = value
    book.save(xlsx)
    e = DE.DatabaseEditor(None, folder, path=xlsx)
    assert e.left_out == ["E0"] and e.modified() and e.windowTitle() == "Database (modified)"
    e.accept()
    first, = V.versions(xlsx)
    assert first.first and "E0" in load_workbook(first.path).worksheets[0][1][column - 1].value
    assert asked == []
    e = DE.DatabaseEditor(None, folder, path=xlsx)
    e.table.item(1, 3).setText("7")
    given += [Box.Save, Box.Cancel]
    e.reject()
    assert [t for t, _ in asked] == ["Close the editor", "Save steels"] and e.result() == 0 and _value(xlsx) == 1


def test_an_import_asks_before_it_is_thrown_away_and_a_new_name_skips_old_versions(folder, answers):
    """An imported table not yet saved: closing asks first. A new database's name skips one whose older versions
    are still there after its file was deleted, so it does not take them on."""
    asked, given = answers
    imported = {"stem": "mine", "names": ["a"], "attributes": ["Density"], "units": ["kg/m^3"], "rows": [["1"]]}
    e = DE.DatabaseEditor(None, folder, imported=imported)
    given.append(Box.Discard)
    e.reject()
    assert asked == [("Close the editor", "mine has changes that aren't saved.")]
    gone = _own(folder)
    V.keep(gone, first=True)
    os.remove(gone)
    os.remove(os.path.splitext(gone)[0] + ".nt")
    imported["stem"] = "steels"
    assert DE.DatabaseEditor(None, folder, imported=imported).name.text() == "steels 2"


def test_materials_offers_versions_for_the_projects_own_database():
    """Versions... in Materials is offered for a database of the project's own, not for an included one, which
    is never changed and so has none."""
    from gui import main_window as G, materials_panel
    d = tempfile.mkdtemp(prefix="_test_gui_", dir=ROOT)
    try:
        w = G.MMTOWindow()
        assert w.open_project(shutil.copy(os.path.join(ROOT, "projects", "U-bracket.mmto"), d))
        assert not materials_panel.MaterialsPanel(w, w.project).versions_button.isEnabled()
        os.makedirs(os.path.join(d, "materials"))
        w.project.database = os.path.relpath(_own(os.path.join(d, "materials")), w.project.base_dir)
        assert materials_panel.MaterialsPanel(w, w.project).versions_button.isEnabled()
    finally:
        shutil.rmtree(d, ignore_errors=True)
