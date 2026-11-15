"""
test_user_guide.py
==================
The user guide: every entry of its lists (selection types, supports and loads, material attributes,
quantities, settings) has a name and a description; docs/user_guide.md holds the lists as the code
makes them now; the guide window opens off-screen, finds a word and marks it on the page; Help >
User guide is in the window, on F1; Help > Researcher guide shows the pages for changing the code in the
same window, a chapter each.

    python tests/docs/test_user_guide.py        (or: python -m pytest tests)
"""
import os
import re
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform.startswith("win"):                         # off-screen Qt on Windows finds no fonts itself
    os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from PyQt5 import QtWidgets  # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from gui import user_guide_window as U  # noqa: E402


def test_every_entry_is_described():
    for name, (header, rows) in U.LISTS.items():
        entries = rows()
        assert entries, f"{name}: empty"
        for entry in entries:
            assert len(entry) == len(header), (name, entry)
            assert str(entry[0]).strip(), f"{name}: an entry with no name: {entry}"
            words = [str(c).strip() for c in entry[1:]]
            assert all(words[i] for i in range(len(words)) if header[i + 1] != "Values"), \
                f"{name}: {entry[0]} has no description"


def test_guide_lists_are_current():
    with open(U.GUIDE, encoding="utf-8") as fh:
        text = fh.read()
    assert text == U.guide_text(), "run python -m gui.user_guide_window --write"
    for name in U.LISTS:
        assert f"<!-- list: {name} -->" in text, name


def test_window_finds_and_marks():
    w = U.UserGuideWindow()
    titles = [title for _, title, _ in w.parts]
    assert any(t.startswith("7. Supports and loads") for t in titles)
    w.search.setText("bolt bores")
    APP.processEvents()
    current = w.contents.currentItem().text(0)
    assert current.startswith("7."), current
    assert w.page.extraSelections(), "no word marked on the page"
    assert w.found.text().startswith("Found in")
    w.search.setText("zzzz not in the guide")
    assert w.found.text() == "Not found"
    w.close()


def test_help_menu_opens_the_guide():
    from gui import main_window as G
    window = G.MMTOWindow()
    actions = [a for m in window.menuBar().actions() if m.menu() for a in m.menu().actions()]
    guide = next(a for a in actions if a.text().replace("&", "") == "User guide")
    assert guide.shortcut().toString() == "F1"
    guide.trigger()
    APP.processEvents()
    assert window._user_guide.isVisible()
    window._user_guide.close()
    window.close()


def test_researcher_guide():
    """Help > Researcher guide, between User guide and About: the five pages for changing the code in the
    guide's window, in their order, each a chapter under its own first heading with its sections as its
    parts; the search finds a word that only CONTRIBUTING.md has."""
    from gui import main_window as G
    window = G.MMTOWindow()
    (menu,) = [m.menu() for m in window.menuBar().actions() if m.menu() and m.text() == "&Help"]
    assert [a.text().replace("&", "") for a in menu.actions()] == ["User guide", "Researcher guide",
                                                                    "About LSR MMTO"]
    assert [os.path.relpath(p, ROOT).replace(os.sep, "/") for p in U.RESEARCHER_GUIDE] == [
        "docs/how_the_code_works.md", "docs/how_to_add_features.md", "docs/code_map.md",
        "docs/reference.md", "CONTRIBUTING.md"]
    menu.actions()[1].trigger()
    APP.processEvents()
    g = window._researcher_guide
    assert g.isVisible() and g.windowTitle() == "Researcher guide" and g is not getattr(window, "_user_guide", None)
    assert g.contents.topLevelItemCount() == 5
    for k, path in enumerate(U.RESEARCHER_GUIDE):
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        chapter = g.contents.topLevelItem(k)
        assert chapter.text(0) == re.search(r"^# (.+)", text, re.M).group(1).strip()
        assert [chapter.child(c).text(0) for c in range(chapter.childCount())] == \
            [t.strip() for t in re.findall(r"^## (.+)", text, re.M)]
    reference = g.contents.topLevelItem(3)
    for title in ("Python", "Settings"):                    # each word is on the page before its heading
        (part,) = [reference.child(c) for c in range(reference.childCount()) if reference.child(c).text(0) == title]
        g.contents.setCurrentItem(part)
        block = g.page.textCursor().block()
        assert block.text() == title and block.blockFormat().headingLevel(), block.text()
    g.contents.setCurrentItem(g.contents.topLevelItem(0))
    assert g.page.toPlainText().startswith("How the code works")
    g.search.setText("lagrangian")                                       # in CONTRIBUTING.md only
    APP.processEvents()
    current = g.contents.currentItem()
    assert (current.parent() or current) is g.contents.topLevelItem(4), current.text(0)
    assert g.page.extraSelections() and g.found.text().startswith("Found in")
    g.close()
    window.close()


def test_a_page_is_laid_out_for_reading():
    """A guide page: every table cell on one line (each column as wide as its longest cell), cells of an empty
    markdown cell kept apart, the header shaded, the text in the middle of each cell; lone asterisks in a
    generated row shown as written, not as italics; the text in a centred column, as wide as the page's widest
    table when that is wider, so a table that fits the window needs no scrolling; a menu path of the window in
    a light box; "section N" a link that opens it, also when it is the part already open; none in a ``` block."""
    from PyQt5 import QtCore, QtGui
    from gui import main_window as G
    window = G.MMTOWindow()
    text = ("## 1. One\n\nUse File > New, then see section 2.\n\n```\nsection 2\n```\n\n"
            + U._table(["name", "units", "objective", "constraint"],
                       [["Compliance", "W/(m*K), mW/(mm*K)", "yes", ""], ["Mass", "kg", "", "yes"]])
            + "\n\n## 2. Two\n\nThe second.\n")
    g = U.UserGuideWindow(window, "Guide", text)
    g.resize(1600, 700)
    g.show()
    APP.processEvents()
    g.contents.setCurrentItem(g.items[0][0])
    APP.processEvents()
    doc = g.page.document()
    (table,) = U._tables(doc)
    assert table.columns() == 4 and all(table.cellAt(r, c).columnSpan() == 1 for r in range(3) for c in range(4))
    assert table.cellAt(1, 2).firstCursorPosition().block().text() == "yes"            # objective, not joined
    assert table.cellAt(2, 3).firstCursorPosition().block().text() == "yes"            # constraint
    assert table.cellAt(1, 1).firstCursorPosition().block().text() == "W/(m*K), mW/(mm*K)"
    for r in range(3):
        for c in range(4):
            cell = table.cellAt(r, c)
            assert cell.format().toTableCellFormat().verticalAlignment() == QtGui.QTextCharFormat.AlignMiddle
            block = cell.firstCursorPosition().block()
            assert block.layout().lineCount() == 1, (r, c, block.text())
    assert table.cellAt(0, 0).format().background().color().name() == U.HEADER
    root = doc.rootFrame().frameFormat()
    assert root.leftMargin() == root.rightMargin() > 22                                  # centred
    assert g.page.horizontalScrollBar().maximum() == 0                                   # it fits: no scroll
    html = g.page.toHtml()
    assert "#section-2" in html and html.count("#section-2") == 1                        # not in the ``` block
    cursor = doc.find("File > New")
    assert cursor.charFormat().background().color().name() == U.MENU
    g.follow(QtCore.QUrl("#section-2"))
    assert g.contents.currentItem().text(0) == "2. Two"
    g.page.verticalScrollBar().setValue(0)
    g.follow(QtCore.QUrl("#section-2"))                                                  # open already
    assert g.page.textCursor().block().text() == "2. Two"
    g.close()
    window.close()


if __name__ == "__main__":
    for f in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        f()
    print("user guide tests passed")
