"""
panel_widgets.py
================
The small widgets and helpers every panel of the LSR MMTO window shares: the number box,
the "i" button, OK and Cancel, colour swatches, and the project's database and its errors.
"""
import ast
import math
import os

from PyQt5 import QtCore, QtGui, QtWidgets

from lsr_mmto.problem.selections import plain_value  # noqa: F401  (the one safe reading of typed text)


GREY = "color: #666666;"


def number(text):
    """A number typed as text, plain arithmetic allowed: 38/2, 0.5*76, 2*pi*5. ValueError otherwise."""
    try:
        value = plain_value(ast.parse(text.strip(), mode="eval").body)
    except SyntaxError:
        raise ValueError(f"{text!r} is not a number") from None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{text!r} is not a number")
    return value


class NumberEdit(QtWidgets.QLineEdit):
    """A number typed as text, shown to 12 significant digits: 1e-07, 0.12345, and 0.0035 for
    the 0.0035000000000000005 that 0.40 * 0.875 * 0.01 gives. A field left as shown hands back the
    stored number exactly; only a typed change replaces it. (PyTO-AD's ScientificDoubleSpinBox
    shows three digits, 0.12345 as 0.12, and hands back what it shows.) With arithmetic=True it also
    takes plain arithmetic (38/2, 2*pi*5) and hands back the number it comes to."""

    def __init__(self, value=0.0, integer=False, parent=None, arithmetic=False):
        super().__init__(parent)
        self.integer, self.arithmetic = integer, arithmetic
        self.set_value(value)
        self.textChanged.connect(lambda _: self.setStyleSheet("" if self.ok() else "background: #ffd6d6;"))

    def set_value(self, value):
        self._exact = int(value) if self.integer else float(value)
        self._shown = str(self._exact) if self.integer else format(self._exact, ".12g")
        self.setText(self._shown)

    def _read(self):
        return number(self.text()) if self.arithmetic else float(self.text())

    def ok(self):
        try:
            v = self._read()
        except ValueError:
            return False
        return math.isfinite(v) and (not self.integer or v == int(v))

    def edited(self):
        """True when the text is no longer what set_value showed (the user typed a change)."""
        return self.text() != self._shown

    def value(self):
        if self.text() == self._shown:
            return self._exact
        v = self._read()
        return int(v) if self.integer else float(v)


CARD_WIDTH = 420               # the (i) card's width; its text wraps inside
CARD_SHARE = 0.6               # a card taller than this share of the window scrolls inside


def card_markdown(text):
    """An (i) text as the card shows it, in Markdown: a blank line starts a paragraph, a single line break
    stays a line break, leading spaces do not make a code block, and a lone * (as in lbf*s^2/in) is a star,
    not emphasis. **bold** and numbered steps (1. ...) are Markdown's own."""
    import re
    lines = [line.strip() for line in text.split("\n")]
    out = []
    for i, line in enumerate(lines):
        line = re.sub(r"(?<=[^\s*])\*(?=[^\s*])", r"\\*", line)
        follows = lines[i + 1] if i + 1 < len(lines) else ""
        hard = line and follows and not re.match(r"(\d+\.|[-*]) ", follows)
        out.append(line + ("  " if hard else ""))
    return "\n".join(out)


def first_sentence(text):
    """What the mouse resting on an (i) shows: its text up to the end of the first sentence, plain."""
    import re
    first = text.strip().split("\n")[0].replace("**", "").strip()
    m = re.search(r"[.!?](\s|$)", first)
    return first[:m.start() + 1] if m else first


class InfoCard(QtWidgets.QFrame):
    """The card an (i) opens: its text in Markdown, CARD_WIDTH wide and wrapped, under the (i) and moved to
    stay inside the window; scrolled inside when taller than CARD_SHARE of the window. A click outside it
    or Esc closes it."""

    def __init__(self, info, text):
        super().__init__(info, QtCore.Qt.Popup)
        self.setObjectName("infoCard")
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.label = QtWidgets.QLabel()
        self.label.setObjectName("infoText")
        self.label.setTextFormat(QtCore.Qt.MarkdownText)
        self.label.setText(card_markdown(text))
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.label.setContentsMargins(14, 10, 14, 10)
        self.label.setFixedWidth(CARD_WIDTH)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidget(self.label)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.addWidget(self.scroll)
        window = info.window().geometry() if info.window() is not None else None
        tall = self.label.heightForWidth(CARD_WIDTH)
        most = int(CARD_SHARE * window.height()) if window is not None else tall
        bar = self.scroll.verticalScrollBar().sizeHint().width() if tall > most else 0
        self.label.setFixedHeight(tall)
        self.setFixedSize(CARD_WIDTH + bar + 2, min(tall, most) + 2)
        self.info, self.window_area = info, window

    def showEvent(self, event):
        super().showEvent(event)
        self.move(self.placed(self.info, self.window_area))

    def placed(self, info, window):
        """Under the (i), its left edge at the (i)'s; moved left and up as far as needed to stay inside the
        window (inside the screen when the (i) has no window)."""
        at = info.mapToGlobal(QtCore.QPoint(0, info.height() + 4))
        area = window if window is not None else QtWidgets.QApplication.primaryScreen().availableGeometry()
        x = max(area.left(), min(at.x(), area.right() - self.width()))
        y = at.y() if at.y() + self.height() <= area.bottom() else info.mapToGlobal(QtCore.QPoint(0, -4)).y() - \
            self.height()
        return QtCore.QPoint(x, max(area.top(), min(y, area.bottom() - self.height())))

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)


class Info(QtWidgets.QToolButton):
    """A small (i): a click opens its card (InfoCard) with the whole text; the mouse resting on it shows the
    first sentence. Its text is its toolTip (setToolTip changes it), written in Markdown: a blank line between
    paragraphs, **bold** lead words, numbered steps."""

    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setText("i")
        self.setToolTip(text)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFixedSize(18, 18)
        self.setStyleSheet("QToolButton { border: 1px solid #8a8a8a; border-radius: 9px; color: #555555; "
                           "font: bold italic 9pt 'Georgia'; background: transparent; } "
                           "QToolButton:hover { background: #e6eefc; }")
        self.card = None
        self.clicked.connect(self.open_card)

    def hover_text(self):
        return first_sentence(self.toolTip())

    def open_card(self):
        self.card = InfoCard(self, self.toolTip())
        self.card.show()
        return self.card

    def event(self, event):
        if event.type() == QtCore.QEvent.ToolTip:
            QtWidgets.QToolTip.showText(event.globalPos(), self.hover_text(), self)
            return True
        return super().event(event)


COLUMN_MOST = 200              # a table's column fits its title and contents up to this width


def fit_columns(table, most=COLUMN_MOST, least=()):
    """Each column of a table as wide as its title, its cells and the widgets in them need, up to most
    pixels (and at least least[c] for the first columns); a title cut short ends in ... and shows in full
    when the mouse rests on it."""
    header = table.horizontalHeader()
    bold = QtGui.QFont(header.font())
    bold.setBold(True)                                          # the headers' titles are bold (app_style.qss);
    header.setFont(bold)                                        # so a title cut short is cut at the right place
    header.setTextElideMode(QtCore.Qt.ElideRight)
    metrics = QtGui.QFontMetrics(bold)
    table.resizeColumnsToContents()
    for c in range(table.columnCount()):
        item = table.horizontalHeaderItem(c)
        title = item.text() if item is not None else ""
        widgets = [table.cellWidget(r, c) for r in range(table.rowCount())]
        want = max([metrics.horizontalAdvance(title) + 28, table.columnWidth(c)] +
                   [w.sizeHint().width() for w in widgets if w is not None] +
                   [least[c] if c < len(least) else 0])
        table.setColumnWidth(c, min(want, most))
        if item is not None:
            item.setToolTip(title if want > most else "")


def with_info(widget, text):
    """widget with an (i) beside it, as one widget."""
    box = QtWidgets.QWidget()
    row = QtWidgets.QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(widget)
    row.addWidget(Info(text))
    row.addStretch()
    return box


def same(a, b):
    """True when a value read back from a widget equals the stored one (1 == 1.0, 'PAPER' == 'PAPER')."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return a == b


def swatch(colour, size=14):
    pix = QtGui.QPixmap(size, size)
    pix.fill(QtGui.QColor(colour))
    return QtGui.QIcon(pix)


def ok_cancel(dialog, layout):
    dialog.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)      # Windows' "?": no help to show
    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    return buttons


def database_errors(project):
    from lsr_mmto.materials.databases import check_database
    xlsx, _ = project.database_paths()
    return check_database(xlsx)[0]


def database_choice(project):
    """(xlsx path or None, error or None) of the project's database."""
    if project.database is None:
        return None, "no database chosen"
    xlsx, _ = project.database_paths()
    return (xlsx, None) if os.path.isfile(xlsx) else (xlsx, f"{xlsx} does not exist")
