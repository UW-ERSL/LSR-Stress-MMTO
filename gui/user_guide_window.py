"""
user_guide_window.py
====================
The user guide inside the window: Help > User guide, or F1. The contents on the left, the page on
the right, and a search box above them. The text is docs/user_guide.md; its lists (selection types,
supports and loads, material attributes, quantities, settings) are made from the code's own lists
when the guide opens, with the same words the forms and their (i) buttons show, so they never fall
behind the code.

    python -m gui.user_guide_window --write     writes the lists into docs/user_guide.md as well,
                                                for reading it outside the window
"""
import os
import re
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

GUIDE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "user_guide.md")
LIST = re.compile(r"(<!-- list: (\w+) -->\n\n)(.*?)(<!-- end of list -->)", re.S)


# ---------------------------------------------------------------------- the lists, from the code
def _row(*cells):
    return "| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in cells) + " |"


def _table(header, rows):
    return "\n".join([_row(*header), _row(*["---"] * len(header))] + [_row(*r) for r in rows])


def selection_types():
    """(name, what it selects) of every selection type the Selection form offers, in its order."""
    from .supports_and_loads_panel import FORM, PICKED
    return [(shown, help) for key, (shown, help, _) in FORM.items() if key not in PICKED] + \
           [(shown, help + " Made with the mouse; open it from Selections to change it.")
            for key, (shown, help, _) in FORM.items() if key in PICKED]


def support_and_load_kinds():
    """(name, support or load, what it does, its values) of every kind, from each physics module."""
    from lsr_mmto.problem.boundary_conditions import BC_KINDS
    return [(k.shown, k.role.capitalize(), k.help, ", ".join(k.labels.get(f[0], f[0]) for f in k.fields))
            for k in BC_KINDS.values()]


def material_attributes():
    from lsr_mmto.materials.databases import ATTRIBUTES
    return [(a.shown, a.doc[0].upper() + a.doc[1:], ", ".join(u for u in a.units if u) or "none")
            for a in ATTRIBUTES.values()]


def quantities():
    from lsr_mmto.formulation.quantities import QUANTITIES
    use = lambda q: " and ".join(w for w, ok in (("objective", q.objective), ("constraint", q.constraint)) if ok)
    return [(q.shown, use(q).capitalize(), q.help) for q in QUANTITIES]


def settings():
    from lsr_mmto.problem.settings import SETTINGS
    return [(s.group, s.label, s.help) for s in SETTINGS]


LISTS = {
    "selection_types": (("Selection", "What it selects"), selection_types),
    "support_and_load_kinds": (("Type", "Support or load", "What it does", "Values"), support_and_load_kinds),
    "material_attributes": (("Attribute", "What it is", "Units"), material_attributes),
    "quantities": (("Quantity", "Can be", "What it is"), quantities),
    "settings": (("Group", "Setting", "What it does"), settings),
}


def list_text(name):
    header, rows = LISTS[name]
    return _table(header, rows()) + "\n\n"


def guide_text(path=GUIDE):
    """docs/user_guide.md with every list made from the code."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    return LIST.sub(lambda m: m.group(1) + list_text(m.group(2)) + m.group(4), text)


def sections(text):
    """[(level, title, markdown)] of the guide: one entry per ## section and per ### part of one."""
    out, current = [], None
    for line in text.splitlines(keepends=True):
        m = re.match(r"(##+) (.+)", line)
        if m and len(m.group(1)) in (2, 3):
            current = [len(m.group(1)), m.group(2).strip(), line]
            out.append(current)
        elif current is not None:
            current[2] += line
    return [tuple(c) for c in out]


# ---------------------------------------------------------------------- the window
class UserGuideWindow(QtWidgets.QDialog):
    """Contents, page and search. A section shows with its parts; a part of a section opens the
    section at that part. The search lists the sections that hold every word typed and marks the
    words on the page."""

    def __init__(self, parent=None, path=GUIDE):
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, True)
        self.setWindowTitle("User guide")
        self.resize(1100, 760)
        self.parts = sections(guide_text(path))
        layout = QtWidgets.QVBoxLayout(self)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search the guide")
        self.search.setClearButtonEnabled(True)
        layout.addWidget(self.search)
        split = QtWidgets.QSplitter()
        self.contents = QtWidgets.QTreeWidget()
        self.contents.setHeaderHidden(True)
        self.page = QtWidgets.QTextBrowser()
        self.page.setOpenExternalLinks(True)
        split.addWidget(self.contents)
        split.addWidget(self.page)
        split.setSizes([300, 800])
        layout.addWidget(split, stretch=1)
        self.found = QtWidgets.QLabel("")
        layout.addWidget(self.found)
        self.items = []                                       # (tree item, index of its ## section)
        top = None
        for i, (level, title, _) in enumerate(self.parts):
            if level == 2:
                top = QtWidgets.QTreeWidgetItem(self.contents, [title])
                self.items.append((top, i))
            elif top is not None:
                self.items.append((QtWidgets.QTreeWidgetItem(top, [title]), i))
        self.contents.currentItemChanged.connect(lambda item, _: self.show_item(item))
        self.search.textChanged.connect(self.find)
        if self.items:
            self.contents.setCurrentItem(self.items[0][0])

    def section_of(self, index):
        """The ## section holding part index, with all its parts: (first index, markdown)."""
        start = index
        while start > 0 and self.parts[start][0] != 2:
            start -= 1
        end = start + 1
        while end < len(self.parts) and self.parts[end][0] != 2:
            end += 1
        return start, "".join(p[2] for p in self.parts[start:end])

    def show_item(self, item):
        index = next((i for it, i in self.items if it is item), None)
        if index is None:
            return
        _, text = self.section_of(index)
        self.page.setMarkdown(text)
        self.page.moveCursor(QtGui.QTextCursor.Start)
        if self.parts[index][0] == 3:                           # a part: the page opens at its heading
            self.page.find(self.parts[index][1])
        words = self.words()
        if words:                                               # and at the first word searched for
            self.page.find(words[0])
        self.mark()

    def words(self):
        return [w for w in self.search.text().lower().split() if w]

    def holds(self, index, words):
        return all(w in self.parts[index][2].lower() for w in words)

    def find(self, *_):
        """Only the parts that hold every word stay in the contents, with their section; the first
        of them opens."""
        words = self.words()
        hits = [(item, self.holds(i, words)) for item, i in self.items]
        for item, hit in hits:
            if item.childCount() == 0:
                item.setHidden(not hit)
        for item, hit in hits:                                  # a section stays when it or a part matched
            if item.childCount():
                below = any(not item.child(c).isHidden() for c in range(item.childCount()))
                item.setHidden(not (hit or below))
                item.setExpanded(bool(words) and below)
        shown = sum(hit for _, hit in hits)
        self.found.setText("" if not words else f"Found in {shown} places" if shown else "Not found")
        first = next((item for item, hit in hits if hit and words), None)
        if first is not None and first is not self.contents.currentItem():
            self.contents.setCurrentItem(first)
        else:
            self.mark()

    def mark(self):
        """Every word of the search marked on the page."""
        marks = []
        doc = self.page.document()
        colour = QtGui.QColor("#ffe08a")
        for w in self.words():
            cursor = doc.find(w)
            while not cursor.isNull():
                sel = QtWidgets.QTextEdit.ExtraSelection()
                sel.cursor = cursor
                sel.format.setBackground(colour)
                marks.append(sel)
                cursor = doc.find(w, cursor)
        self.page.setExtraSelections(marks)


def open_user_guide(window):
    """The guide, one window kept open beside the main window: Help > User guide and F1."""
    guide = getattr(window, "_user_guide", None)
    if guide is None:
        guide = window._user_guide = UserGuideWindow(window)
    guide.show()
    guide.raise_()
    guide.activateWindow()
    return guide


def write_lists(path=GUIDE):
    """docs/user_guide.md with its lists made from the code, for reading it outside the window."""
    text = guide_text(path)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


if __name__ == "__main__" and "--write" in sys.argv:
    write_lists()
