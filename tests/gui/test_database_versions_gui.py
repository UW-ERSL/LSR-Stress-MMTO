"""
test_database_versions_gui.py
=============================
The versions of a database in the window, off-screen: the editor's (modified) title and its one question
(changes not saved, on closing); every Save kept as a version without asking, for a database made, imported or
copied from an included one too; Versions... with "(in use)", Restore without a question, Rename... of a note
and Delete (never the one in use); and in Materials: Rename..., Import copying a database in, and a change made
in Excel kept as Materials opens. The questions are answered by the test (database_editor.ask), never shown.

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
    editor open, Discard changes closes it with the file as it was; Save... saves, keeping v1 (as it was) and v2
    (as saved)."""
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
    given.append(Box.Save)
    e.reject()
    assert e.result() == QtWidgets.QDialog.Accepted and _value(xlsx) == 7
    assert [(v.number, _value(v.path)) for v in V.versions(xlsx)] == [(2, 7), (1, 1)]
    e = DE.DatabaseEditor(None, folder, path=xlsx)
    e.reject()                                                 # nothing changed: nothing asked
    assert len(asked) == 3 and e.result() == QtWidgets.QDialog.Rejected


def test_every_save_is_a_version_without_asking(folder, answers):
    """Each Save that changes the database keeps it as saved as the next version, asking nothing; the state it
    had before is v1. A Save with nothing changed keeps nothing new."""
    asked, given = answers
    xlsx = _own(folder)
    for value in (7, 8, 9):
        e = DE.DatabaseEditor(None, folder, path=xlsx)
        e.table.item(1, 3).setText(str(value))
        e.accept()
    assert asked == [] and _value(xlsx) == 9
    assert [(v.number, _value(v.path)) for v in V.versions(xlsx)] == [(4, 9), (3, 8), (2, 7), (1, 1)]
    assert V.in_use(xlsx).number == 4
    e = DE.DatabaseEditor(None, folder, path=xlsx)
    e.accept()
    assert len(V.versions(xlsx)) == 4


def test_a_new_database_and_a_copy_of_an_included_one_get_their_versions(folder, answers):
    """Saved as new (imported here), its v1 is itself; an included database edited, saved as a copy in the
    folder: the copy's v1 is the included database as it came, with its network, and v2 the copy as saved."""
    asked, given = answers
    imported = {"stem": "mine", "names": ["a", "b"], "attributes": ["Density", "Youngs_Modulus"],
                "units": ["kg/m^3", "Pa"], "rows": [["1", "100"], ["2", "200"]]}
    e = DE.DatabaseEditor(None, folder, imported=imported)
    e.accept()
    path = os.path.join(folder, "mine.xlsx")
    assert e.result() == QtWidgets.QDialog.Accepted and [v.number for v in V.versions(path)] == [1]
    e = DE.DatabaseEditor(None, folder, path=DATABASE + ".xlsx")
    e.table.item(1, 3).setText("7")
    e.accept()
    copy = e.saved_path
    assert os.path.dirname(copy) == folder and _value(copy) == 7
    v2, v1 = V.versions(copy)
    assert _value(v1.path) == 1 and os.path.isfile(v1.network) and _value(v2.path) == 7
    assert V.in_use(copy) == v2 and asked == []
    assert not os.path.isdir(os.path.join(os.path.dirname(DATABASE), "versions"))   # the included one: untouched


def test_versions_marks_the_one_in_use_restores_renames_and_deletes(folder, answers):
    """Versions...: the versions newest first, "(in use)" beside the database's own. Restore asks nothing, keeps
    every version and closes the window, the restored one now in use; the one in use can't be deleted or
    restored; Rename... gives a note; Delete asks first, and No keeps the version."""
    from gui.database_versions_window import VersionsWindow
    asked, given = answers
    xlsx = _own(folder)
    e = DE.DatabaseEditor(None, folder, path=xlsx)
    e.table.item(1, 3).setText("7")
    e.accept()
    w = VersionsWindow(None, xlsx)
    assert w.list.count() == 2 and w.list.item(0).text().startswith("steels v2 (in use)")
    assert "6 materials" in w.list.item(1).text() and "(in use)" not in w.list.item(1).text()
    assert not w.delete_button.isEnabled() and not w.restore_button.isEnabled()
    assert w.rename("softer") and w.list.item(0).text().startswith("steels v2 softer (in use)")
    w.list.setCurrentRow(1)
    assert w.restore() and w.result() == QtWidgets.QDialog.Accepted and w.restored.number == 1
    assert _value(xlsx) == 1 and asked == [] and len(V.versions(xlsx)) == 2
    w = VersionsWindow(None, xlsx)
    assert w.list.item(1).text().startswith("steels v1 (in use)")
    w.list.setCurrentRow(0)
    given.append(Box.No)
    assert not w.delete() and w.list.count() == 2
    assert asked[-1] == ("Delete version", "Delete steels v2 softer? This can't be undone.")
    given.append(Box.Yes)
    assert w.delete() and w.list.count() == 1


def test_a_save_that_changes_the_file_keeps_it_as_it_was_first(folder, answers):
    """A database with a column LSR MMTO does not know: the editor shows (modified) from the start, since Save
    writes it without that column, and that Save keeps the file as it was (v1, with the column) and as saved
    (v2). Save... from the close question saves at once."""
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
    v2, v1 = V.versions(xlsx)
    assert "E0" in load_workbook(v1.path).worksheets[0][1][column - 1].value and asked == []
    e = DE.DatabaseEditor(None, folder, path=xlsx)
    e.table.item(1, 3).setText("7")
    given.append(Box.Save)
    e.reject()
    assert [t for t, _ in asked] == ["Close the editor"] and _value(xlsx) == 7 and len(V.versions(xlsx)) == 3


def test_an_import_asks_before_it_is_thrown_away_and_a_new_name_skips_old_versions(folder, answers):
    """An imported table not yet saved: closing asks first. A new database's name skips one whose versions are
    still there after its file was deleted, so it does not take them on."""
    asked, given = answers
    imported = {"stem": "mine", "names": ["a"], "attributes": ["Density"], "units": ["kg/m^3"], "rows": [["1"]]}
    e = DE.DatabaseEditor(None, folder, imported=imported)
    given.append(Box.Discard)
    e.reject()
    assert asked == [("Close the editor", "mine has changes that aren't saved.")]
    gone = _own(folder)
    V.record(gone)
    os.remove(gone)
    os.remove(os.path.splitext(gone)[0] + ".nt")
    imported["stem"] = "steels"
    assert DE.DatabaseEditor(None, folder, imported=imported).name.text() == "steels 2"


def test_materials_keeps_versions_renames_imports_and_says_what_was_restored(answers):
    """In Materials: Versions... and Rename... only for a database of the project's own; opening it keeps the
    database as a version, and a change made in Excel since as the next; Rename... renames the database, its
    network and versions, and saves the project with the new name; Import copies a database of LSR MMTO into the
    project's folder; a restore says which version is now used."""
    from gui import main_window as G, materials_panel
    from gui.database_versions_window import VersionsWindow
    from lsr_mmto.problem.project import Project
    d = tempfile.mkdtemp(prefix="_test_gui_", dir=ROOT)
    try:
        w = G.MMTOWindow()
        assert w.open_project(shutil.copy(os.path.join(ROOT, "projects", "U-bracket.mmto"), d))
        panel = materials_panel.MaterialsPanel(w, w.project)
        assert not panel.versions_button.isEnabled() and not panel.rename_button.isEnabled()
        assert w.save_project_as(os.path.join(d, "mine", "mine.mmto"))
        materials = os.path.join(w.project.base_dir, "materials")
        os.makedirs(materials, exist_ok=True)
        w.project.database = os.path.relpath(_own(materials), w.project.base_dir).replace("\\", "/")
        w._write(w.path)
        panel = materials_panel.MaterialsPanel(w, w.project)
        assert panel.versions_button.isEnabled() and panel.rename_button.isEnabled()
        xlsx = os.path.join(materials, "steels.xlsx")
        assert [v.number for v in V.versions(xlsx)] == [1]
        from lsr_mmto.materials.databases import database_table, material_display, write_database
        names, attributes, units, values = database_table(xlsx)
        values[0, 0] = 7.0
        write_database(xlsx, attributes, units, names, material_display(xlsx)[1], material_display(xlsx)[0],
                       values.tolist())                         # as Excel would change it
        panel = materials_panel.MaterialsPanel(w, w.project)
        assert [v.number for v in V.versions(xlsx)] == [2, 1] and V.in_use(xlsx).number == 2
        assert panel.rename_database("alloys") == "materials/alloys.xlsx"
        renamed = os.path.join(materials, "alloys.xlsx")
        assert os.path.isfile(renamed) and not os.path.exists(xlsx) and [v.number for v in V.versions(renamed)] == [2, 1]
        assert Project.load(w.path).database == "materials/alloys.xlsx"
        real = VersionsWindow.exec_
        VersionsWindow.exec_ = lambda self: (self.list.setCurrentRow(1), self.restore())
        try:
            panel.open_versions()
        finally:
            VersionsWindow.exec_ = real
        assert panel.report.toPlainText().startswith("Now using alloys v1.") and _value(renamed) == 1
        outside = tempfile.mkdtemp()
        try:
            shutil.copy(DATABASE + ".xlsx", os.path.join(outside, "alloys.xlsx"))
            copied = panel.import_database(os.path.join(outside, "alloys.xlsx"))
        finally:
            shutil.rmtree(outside, ignore_errors=True)
        assert copied == os.path.join(materials, "alloys 2.xlsx") and os.path.isfile(copied)
    finally:
        shutil.rmtree(d, ignore_errors=True)
