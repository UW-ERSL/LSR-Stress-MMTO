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


class Info(QtWidgets.QToolButton):
    """A small (i): the explanation shows on a click, or when the mouse rests on it."""

    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setText("i")
        self.setToolTip(text)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFixedSize(18, 18)
        self.setStyleSheet("QToolButton { border: 1px solid #8a8a8a; border-radius: 9px; color: #555555; "
                           "font: bold italic 9pt 'Georgia'; background: transparent; } "
                           "QToolButton:hover { background: #e6eefc; }")
        self.clicked.connect(lambda: QtWidgets.QToolTip.showText(
            self.mapToGlobal(self.rect().bottomLeft()), self.toolTip(), self))


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
