"""
user_guide_window.py
====================
The user guide inside the window: Help > User guide, or F1. The contents on the left, the page on
the right, and a search box above them. The text is docs/user_guide.md; its lists (selection types,
supports and loads, material attributes, quantities, settings) are made from the code's own lists
when the guide opens, with the same words the forms and their (i) buttons show, so they never fall
behind the code.

Help > Researcher guide shows, in the same window, the pages for changing the code (RESEARCHER_GUIDE,
read from their files when it opens), one chapter each.

A page (GuidePage) is laid out for reading: the text in a centred column of a readable width (READING_WIDTH),
space between lines and paragraphs, code names in a code font on grey, a menu path (File > New) in a light box,
"section 6.3" a link to it, and tables with thin lines and a shaded header, every cell on one line and its text
in the middle (a table wider than the page scrolls sideways). Markdown joins an empty cell to the one before
it; the page splits them again, so each value stays in its own column.

    python -m gui.user_guide_window --write     writes the lists into docs/user_guide.md as well,
                                                for reading it outside the window
"""
import os
import re
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUIDE = os.path.join(ROOT, "docs", "user_guide.md")
RESEARCHER_GUIDE = [os.path.join(ROOT, f) for f in ("docs/how_the_code_works.md", "docs/how_to_add_features.md",
                                                  "docs/code_map.md", "docs/reference.md",
                                                  "CONTRIBUTING.md")]           # its chapters, in this order
LIST = re.compile(r"(<!-- list: (\w+) -->\n\n)(.*?)(<!-- end of list -->)", re.S)


# ---------------------------------------------------------------------- the lists, from the code
def _row(*cells):
    """A table row; a lone * (lbf*s^2/in) escaped, so markdown shows it, not italics."""
    return "| " + " | ".join(re.sub(r"(?<![\\*])\*(?!\*)", r"\\*", str(c).replace("|", "/").replace("\n", " "))
                             for c in cells) + " |"


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
    return [(k.shown, k.role.capitalize(), k.help, ", ".join(v.label for v in k.values))
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


def chapters_text(paths=RESEARCHER_GUIDE):
    """The files as one text, each a chapter: its # title a ## section, its ## parts ### parts (every heading
    one level down)."""
    text = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            text.append(re.sub(r"^(#+) ", r"#\1 ", fh.read(), flags=re.M))
    return "\n\n".join(text)


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


# ---------------------------------------------------------------------- the page
READING_WIDTH = 760                                         # the text column, in pixels: about 90 characters
LINE = "#dde2ea"                                            # a table's lines
HEADER = "#eef2f8"                                          # a table's header row
CODE = "#f1f3f6"                                            # behind a code name
MENU = "#e8edf7"                                            # behind a menu path
CODE_FONTS = ("Cascadia Mono", "Consolas", "DejaVu Sans Mono", "Menlo")
SECTION_LINK = re.compile(r"\bsection (\d+(?:\.\d+)*)")


def section_number(title):
    """The number a section's title starts with ("6.3 The database editor": "6.3"), or None."""
    m = re.match(r"(\d+(?:\.\d+)*)\.?\s", title)
    return m.group(1) if m else None


def linked(markdown, numbers):
    """markdown with each "section N" that names a section of numbers a link to it (#section-N); none in a
    heading, in `code` or in a ``` block."""
    out, fenced = [], False
    for line in markdown.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if fenced or line.lstrip().startswith("```") or line.startswith("#"):
            out.append(line)
            continue
        parts = re.split(r"(`[^`]*`)", line)
        out.append("".join(p if p.startswith("`") else SECTION_LINK.sub(
            lambda m: f"[section {m.group(1)}](#section-{m.group(1)})" if m.group(1) in numbers else m.group(0), p)
            for p in parts))
    return "".join(out)


def _frames(frame):
    for child in frame.childFrames():
        yield child
        yield from _frames(child)


def _tables(doc):
    return [f for f in _frames(doc.rootFrame()) if isinstance(f, QtGui.QTextTable)]


class GuidePage(QtWidgets.QTextBrowser):
    """A page of a guide, laid out for reading (see the module's docstring). menu_paths: the menu paths of the
    window ("File > New"), to be shown in a light box."""

    def __init__(self, menu_paths=()):
        super().__init__()
        self.menu_paths = sorted(menu_paths, key=len, reverse=True)
        self.widest = 0                                     # the page's widest table, in pixels

    def show_markdown(self, text):
        """The page laid out from markdown text. It is laid out in a document of its own and shown when done:
        each change to a document already on screen would lay the page out again."""
        doc = QtGui.QTextDocument(self)
        doc.setDefaultFont(self.document().defaultFont())
        doc.setDocumentMargin(self.document().documentMargin())
        doc.setMarkdown(text)
        option = doc.defaultTextOption()
        option.setWrapMode(QtGui.QTextOption.WordWrap)        # never inside a word
        doc.setDefaultTextOption(option)
        for table in _tables(doc):
            self._split_joined(table)
        self._space(doc)
        self._code_and_menus(doc)
        self._tables_look(doc)
        self._table_widths(doc)
        self.setDocument(doc)                                 # the page's last document is deleted with it
        self.fit()

    @staticmethod
    def _split_joined(table):
        """Cells markdown joined (an empty cell joins the one before it) split again."""
        for r in range(table.rows()):
            for c in range(table.columns()):
                cell = table.cellAt(r, c)
                if cell.column() == c and cell.row() == r and (cell.columnSpan() > 1 or cell.rowSpan() > 1):
                    table.splitCell(r, c, 1, 1)

    def _space(self, doc):
        """Room between lines and paragraphs, and more above headings; table cells keep their own."""
        block = doc.begin()
        while block.isValid():
            if QtGui.QTextCursor(block).currentTable() is None:
                fmt = block.blockFormat()
                level = fmt.headingLevel()
                fmt.setLineHeight(120 if level else 145, QtGui.QTextBlockFormat.ProportionalHeight)
                fmt.setTopMargin(22 if level == 2 else 16 if level else fmt.topMargin())
                fmt.setBottomMargin(8 if level else 12)
                QtGui.QTextCursor(block).setBlockFormat(fmt)
            block = block.next()

    def _code_and_menus(self, doc):
        """Code names in a code font on grey; each menu path of the window in a light box."""
        have = set(QtGui.QFontDatabase().families())
        mono = next((f for f in CODE_FONTS if f in have),
                    QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont).family())
        code = QtGui.QTextCharFormat()
        code.setFontFamily(mono)
        code.setBackground(QtGui.QColor(CODE))
        menu = QtGui.QTextCharFormat()
        menu.setBackground(QtGui.QColor(MENU))
        paths = re.compile("|".join(re.escape(p) for p in self.menu_paths)) if self.menu_paths else None
        block = doc.begin()
        while block.isValid():
            spans = []
            it = block.begin()
            while not it.atEnd():
                fragment = it.fragment()
                if fragment.isValid() and fragment.charFormat().fontFixedPitch():
                    spans.append((fragment.position(), fragment.length(), code))
                it += 1
            if paths is not None:
                spans += [(block.position() + m.start(), m.end() - m.start(), menu)
                          for m in paths.finditer(block.text())]
            for start, length, fmt in spans:
                cursor = QtGui.QTextCursor(doc)
                cursor.setPosition(start)
                cursor.setPosition(start + length, QtGui.QTextCursor.KeepAnchor)
                cursor.mergeCharFormat(fmt)
            block = block.next()

    def _tables_look(self, doc):
        """Thin lines, a shaded header, room in each cell, its text in the middle."""
        for table in _tables(doc):
            fmt = table.format()
            fmt.setBorderCollapse(True)
            fmt.setBorder(1)
            fmt.setBorderBrush(QtGui.QBrush(QtGui.QColor(LINE)))
            fmt.setBorderStyle(QtGui.QTextFrameFormat.BorderStyle_Solid)
            fmt.setCellSpacing(0)
            fmt.setCellPadding(0)
            fmt.setTopMargin(10)
            fmt.setBottomMargin(18)
            table.setFormat(fmt)
            for r in range(table.rows()):
                for c in range(table.columns()):
                    cell = table.cellAt(r, c)
                    cf = cell.format().toTableCellFormat()
                    cf.setVerticalAlignment(QtGui.QTextCharFormat.AlignMiddle)
                    cf.setLeftPadding(12)
                    cf.setRightPadding(12)
                    cf.setTopPadding(8)
                    cf.setBottomPadding(8)
                    if r == 0:
                        cf.setBackground(QtGui.QColor(HEADER))
                    cell.setFormat(cf)

    def fit(self):
        """The text column centred at READING_WIDTH, or as wide as the page's widest table, so a table that
        fits the window needs no scrolling; the page's whole width when that is narrower."""
        doc = self.document()
        side = max(22, (self.viewport().width() - max(READING_WIDTH, self.widest)) // 2)
        root = doc.rootFrame().frameFormat()
        if (root.leftMargin(), root.rightMargin()) != (side, side):
            root.setLeftMargin(side)
            root.setRightMargin(side)
            doc.rootFrame().setFrameFormat(root)

    def _table_widths(self, doc):
        """Each table's columns as wide as their longest cell, so no cell wraps; widest, the widest table."""
        self.widest = 0
        for table in _tables(doc):
            widths = []
            for c in range(table.columns()):
                widest = 0
                for r in range(table.rows()):
                    cell = table.cellAt(r, c)
                    end = cell.lastCursorPosition().position()
                    block, width = doc.findBlock(cell.firstCursorPosition().position()), 0.0
                    while block.isValid() and block.position() <= end:
                        it = block.begin()
                        while not it.atEnd():
                            fragment = it.fragment()
                            font = fragment.charFormat().font().resolve(doc.defaultFont())
                            width += QtGui.QFontMetricsF(font).horizontalAdvance(fragment.text())
                            it += 1
                        block = block.next()
                    widest = max(widest, width)
                widths.append(QtGui.QTextLength(QtGui.QTextLength.FixedLength, widest + 34))  # its padding too
            fmt = table.format()
            fmt.setColumnWidthConstraints(widths)
            table.setFormat(fmt)
            self.widest = max(self.widest, int(sum(w.rawValue() for w in widths)) + 4)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit()


def menu_paths(window):
    """The window's menu paths, as the guides write them: "File > New", "Help > User guide"."""
    if window is None or not hasattr(window, "menuBar"):
        return []
    out = []
    for top in window.menuBar().actions():
        menu = top.menu()
        if menu is None:
            continue
        name = top.text().replace("&", "")
        out += [f"{name} > {a.text().replace('&', '').rstrip('.').strip()}" for a in menu.actions()
                if a.text() and not a.isSeparator()]
    return out


# ---------------------------------------------------------------------- the window
class UserGuideWindow(QtWidgets.QDialog):
    """Contents, page and search. A section shows with its parts; a part of a section opens the
    section at that part. The search lists the sections that hold every word typed and marks the
    words on the page."""

    def __init__(self, parent=None, title="User guide", text=None):
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, True)
        self.setWindowTitle(title)
        self.resize(1100, 760)
        self.parts = sections(guide_text() if text is None else text)
        layout = QtWidgets.QVBoxLayout(self)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search the guide")
        self.search.setClearButtonEnabled(True)
        layout.addWidget(self.search)
        split = QtWidgets.QSplitter()
        self.contents = QtWidgets.QTreeWidget()
        self.contents.setHeaderHidden(True)
        self.page = GuidePage(menu_paths(parent))
        self.page.setOpenLinks(False)                           # "section 6.3" here; a web address outside
        self.page.anchorClicked.connect(self.follow)
        split.addWidget(self.contents)
        split.addWidget(self.page)
        split.setSizes([260, 840])
        layout.addWidget(split, stretch=1)
        self.found = QtWidgets.QLabel("")
        layout.addWidget(self.found)
        self.numbers = {n: i for i, (_, title, _) in enumerate(self.parts)
                        if (n := section_number(title)) is not None}   # "6.3": its part's index
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
        self.page.show_markdown(linked(text, self.numbers))
        self.page.moveCursor(QtGui.QTextCursor.Start)
        if self.parts[index][0] == 3:                           # a part: the page opens at its heading
            self.at_heading(self.parts[index][1])
        words = self.words()
        if words:                                               # and at the first word searched for
            self.page.find(words[0])
        self.mark()

    def follow(self, url):
        """A link clicked: "section N" opens that section at its heading; a web address opens outside."""
        fragment = url.fragment()
        if fragment.startswith("section-") and fragment[len("section-"):] in self.numbers:
            index = self.numbers[fragment[len("section-"):]]
            item = next((it for it, i in self.items if i == index), None)
            if item is self.contents.currentItem():
                self.show_item(item)                            # already open: back to its heading
            elif item is not None:
                self.contents.setCurrentItem(item)
            return
        if url.scheme() in ("http", "https", "mailto"):
            QtGui.QDesktopServices.openUrl(url)

    def at_heading(self, title):
        """The page scrolled to the heading with this text, not to the same word earlier on the page."""
        block = self.page.document().begin()
        while block.isValid() and not (block.blockFormat().headingLevel() and block.text().strip() == title):
            block = block.next()
        if block.isValid():
            self.page.setTextCursor(QtGui.QTextCursor(block))
            top = self.page.document().documentLayout().blockBoundingRect(block).top()
            self.page.verticalScrollBar().setValue(int(top))

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
    return _kept_open(window, "_user_guide", lambda: UserGuideWindow(window))


def open_researcher_guide(window):
    """The pages for changing the code, one window kept open beside the main window: Help > Researcher guide."""
    return _kept_open(window, "_researcher_guide", lambda: UserGuideWindow(window, "Researcher guide", chapters_text()))


def _kept_open(window, attr, make):
    guide = getattr(window, attr, None)
    if guide is None:
        guide = make()
        setattr(window, attr, guide)
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
