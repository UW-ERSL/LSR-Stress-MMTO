"""
test_user_guide.py
==================
The user guide: every entry of its lists (selection types, supports and loads, material attributes,
quantities, settings) has a name and a description; docs/user_guide.md holds the lists as the code
makes them now; the guide window opens off-screen, finds a word and marks it on the page; Help >
User guide is in the window, on F1.

    python tests/test_user_guide.py        (or: python -m pytest tests)
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform.startswith("win"):                         # off-screen Qt on Windows finds no fonts itself
    os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


if __name__ == "__main__":
    for f in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        f()
    print("user guide tests passed")
