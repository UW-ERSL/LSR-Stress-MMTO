"""
geometry_panel.py
=================
The Geometry strip, the first step of a project: its part, from an STL file or made of shapes. On the shapes
side, the list of steps, and under it the boxes of the step chosen; every choice is a switch or a tile, never a
drop-down list. The strip only edits the project's steps with the functions of lsr_mmto/problem/shapes.py; the
part is built (part_from_shapes) and drawn only on Update part, and on closing the strip.

    window.open_geometry_panel()        # docked on the right, like Supports and loads
"""
import math
import os

from PyQt5 import QtCore, QtWidgets

from lsr_mmto.problem import shapes as S
from lsr_mmto.problem.shapes import ShapeError

from .panel_widgets import GREY, Info, NumberEdit, number, set_style

GEOMETRY_HELP = (
    "**The part** is what is optimized: an STL file made in a CAD program, or a part made here from shapes.\n\n"
    "**From shapes**, the part is a list of steps, done in order. Each step adds a shape to the part, cuts it out "
    "of the part, or keeps only the overlap of the two. The first step adds.\n\n"
    "**At** is where the shape's origin goes: a box's lowest corner, the centre of a cylinder's base, a sphere's "
    "centre. **Turn** turns the shape about the x, y and z axes through its origin, in that order, before it is "
    "moved. **Repeat** makes copies along a line (each moved by the Spacing from the one before) or around a "
    "circle; **Count** is how many, the first one included.\n\n"
    "**Update part** builds the part from the steps and draws it. The user guide (section 5.2) explains every "
    "box.")

TILES = (("box", "Box"), ("cylinder", "Cylinder"), ("sphere", "Sphere"), ("extrusion", "Extrusion"),
         ("revolution", "Revolution"), ("wrap", "Wrap"))
WRAPPED_TILES = TILES[:4]                                         # what a wrap is made of
RED = "color: #b00000;"
WAITING = "The view shows the part before your last changes."


def default_shape(kind, size):
    """A shape of this kind about size across, to start from."""
    if kind == "box":
        return S.box(size, size, size)
    if kind == "cylinder":
        return S.cylinder(size / 2, size)
    if kind == "sphere":
        return S.sphere(size / 2)
    if kind == "extrusion":
        return S.extrusion(S.rectangle(size, size), size)
    if kind == "revolution":
        return S.revolution(S.rectangle(size / 2, size))
    return S.wrap(S.place(S.cylinder(size / 4, size / 4)), S.place(S.cylinder(size / 4, size / 4), at=(size, 0, 0)))


def nice(value):
    """A round number near value (1, 2 or 5 times a power of ten) for a shape to start from."""
    if value <= 0 or not math.isfinite(value):
        return 10.0
    k = math.floor(math.log10(value))
    return float(min((1, 2, 5, 10), key=lambda d: abs(d * 10 ** k - value)) * 10 ** k)


def line_of(step):
    """A step as its line in the list: the step in words (shapes.describe), its action first: "Cut cylinder ..."."""
    text = S.describe(step)
    return text[0].upper() + text[1:]


def points_from_text(text):
    """The (x, y) points in text pasted from Excel or a text file: two numbers a row, split by tabs, commas or
    spaces; a first row with no number (a header) is skipped. ValueError naming the first row that is not two
    numbers."""
    rows = [r for r in text.replace("\r", "").split("\n") if r.strip()]
    points = []
    for k, row in enumerate(rows, 1):
        cells = [c for c in row.replace("\t", " ").replace(",", " ").replace(";", " ").split() if c]
        try:
            values = [number(c) for c in cells]
        except ValueError:
            if k == 1 and not any(ch.isdigit() for ch in row):
                continue                                            # a header
            raise ValueError(f"row {k} ({row.strip()!r}) is not two numbers") from None
        if len(values) != 2:
            raise ValueError(f"row {k} ({row.strip()!r}) is not two numbers")
        points.append([float(values[0]), float(values[1])])
    if len(points) < 3:
        raise ValueError("a polygon needs at least 3 points")
    return points


class Switch(QtWidgets.QWidget):
    """A row of buttons of which one is chosen (a segmented switch): every choice on screen, no list to open.
    changed(value) on a click."""

    changed = QtCore.pyqtSignal(str)

    def __init__(self, choices, value=None, parent=None):
        super().__init__(parent)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.buttons = {}
        group = QtWidgets.QButtonGroup(self)
        for key, text in choices:
            b = QtWidgets.QToolButton()
            b.setText(text)
            b.setObjectName("segment")
            b.setCheckable(True)
            b.setCursor(QtCore.Qt.PointingHandCursor)
            group.addButton(b)
            row.addWidget(b)
            b.clicked.connect(lambda _=False, k=key: self._clicked(k))
            self.buttons[key] = b
        row.addStretch()
        self.set(value if value is not None else choices[0][0])

    def _clicked(self, key):
        if key != self._last:                                       # the choice already made: nothing changes
            self._last = key
            self.changed.emit(key)

    def set(self, value):
        """value chosen, without changed()."""
        self.buttons[value].setChecked(True)
        self._last = value

    def value(self):
        """The choice made."""
        return self._last


def emptied(layout):
    """Every widget of layout taken out, hidden at once and deleted (by Qt, when the app is next idle): boxes
    made again on each change never stay on screen beside their new ones, even for that moment."""
    while layout.count():
        w = layout.takeAt(0).widget()
        if w is not None:
            w.hide()
            w.deleteLater()


def link(text, slot):
    """A button drawn as a blue word (+ Turn, Remove repeat); slot() on a click."""
    b = QtWidgets.QToolButton()
    b.setText(text)
    b.setObjectName("link")
    b.setCursor(QtCore.Qt.PointingHandCursor)
    b.clicked.connect(lambda _=False: slot())
    return b


def tiles(choices, slot):
    """The shapes as tiles, three to a row; slot(kind) on a click."""
    box = QtWidgets.QWidget()
    grid = QtWidgets.QGridLayout(box)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setSpacing(6)
    for k, (kind, text) in enumerate(choices):
        b = QtWidgets.QToolButton()
        b.setText(text)
        b.setObjectName("tile")
        b.setCursor(QtCore.Qt.PointingHandCursor)
        b.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        b.clicked.connect(lambda _=False, kd=kind: slot(kd))
        grid.addWidget(b, k // 3, k % 3)
    return box


class Fields(QtWidgets.QWidget):
    """Labelled number boxes in rows; edited() after a box is finished (Enter, Tab or a click elsewhere) holding
    a number. value(key) is its number."""

    edited = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.form = QtWidgets.QGridLayout(self)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setHorizontalSpacing(6)
        self.form.setVerticalSpacing(4)
        self.boxes = {}

    def row(self, label, keys, values, unit="", integer=False):
        """A row: its label, a box per key (several for x, y, z), and the unit after them."""
        r = self.form.rowCount()
        self.form.addWidget(QtWidgets.QLabel(label), r, 0)
        for c, (key, value) in enumerate(zip(keys, values), 1):
            box = NumberEdit(value, integer=integer, arithmetic=True)
            box.setMinimumWidth(48)
            box.editingFinished.connect(lambda b=box: self._finished(b))
            self.form.addWidget(box, r, c)
            self.boxes[key] = box
        if unit:
            u = QtWidgets.QLabel(unit)
            set_style(u, GREY)
            self.form.addWidget(u, r, len(keys) + 1)

    def _finished(self, box):
        if box.ok() and box.edited():
            box.set_value(box.value())                              # taken: shown as kept
            self.edited.emit()

    def value(self, key):
        return self.boxes[key].value()

    def values(self, *keys):
        return [self.value(k) for k in keys]

    def all_numbers(self):
        """Whether every box holds a number (a box with other text is red, and nothing is taken till it does)."""
        return all(b.ok() for b in self.boxes.values())


class OutlineEditor(QtWidgets.QWidget):
    """An outline's kind (a switch) and its boxes: a rectangle's width, depth and round, a circle's radius, a
    polygon's points (a table, with Paste points) and round. changed() after an edit; outline() is the outline."""

    changed = QtCore.pyqtSignal()

    def __init__(self, outline, unit, circle=True, parent=None):
        super().__init__(parent)
        self.outline_ = dict(outline)
        self.unit = unit
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        choices = [("rectangle", "Rectangle")] + ([("circle", "Circle")] if circle else []) + [("polygon", "Polygon")]
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Outline"))
        self.kind = Switch(choices, outline["kind"])
        self.kind.changed.connect(self._kind_changed)
        top.addWidget(self.kind, stretch=1)
        lay.addLayout(top)
        self.body = QtWidgets.QVBoxLayout()
        lay.addLayout(self.body)
        self.error = QtWidgets.QLabel("")
        self.error.setWordWrap(True)
        set_style(self.error, RED)
        self.error.hide()
        lay.addWidget(self.error)
        self._fill()

    def _clear(self):
        emptied(self.body)

    def _fill(self):
        self._clear()
        o = self.outline_
        self.fields = Fields()
        if o["kind"] == "rectangle":
            self.fields.row("Width", ["width"], [o["width"]], self.unit)
            self.fields.row("Depth", ["depth"], [o["depth"]], self.unit)
            self.fields.row("Corner radius", ["round"], [o.get("round", 0)], self.unit)
        elif o["kind"] == "circle":
            self.fields.row("Radius", ["radius"], [o["radius"]], self.unit)
        else:
            self.fields.row("Corner radius", ["round"], [o.get("round", 0)], self.unit)
        self.fields.edited.connect(self._edited)
        self.body.addWidget(self.fields)
        self.table = None
        if o["kind"] == "polygon":
            self.table = QtWidgets.QTableWidget(len(o["points"]), 2)
            self.table.setHorizontalHeaderLabels(["x", "y"])
            self.table.verticalHeader().setDefaultSectionSize(22)
            self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
            self.table.setMinimumHeight(120)
            for r, (x, y) in enumerate(o["points"]):
                for c, v in enumerate((x, y)):
                    self.table.setItem(r, c, QtWidgets.QTableWidgetItem(format(v, ".12g")))
            self.table.itemChanged.connect(lambda _: self._edited())
            self.body.addWidget(self.table)
            row = QtWidgets.QHBoxLayout()
            for text, slot in (("Add point", self._add_point), ("Remove point", self._remove_point),
                               ("Paste points", self.paste_points)):
                b = QtWidgets.QPushButton(text)
                b.clicked.connect(lambda _=False, s=slot: s())
                row.addWidget(b)
            row.addStretch()
            holder = QtWidgets.QWidget()
            holder.setLayout(row)
            self.body.addWidget(holder)

    def _kind_changed(self, kind):
        o = self.outline_
        size = o.get("width") or 2 * o.get("radius", 0) or 10.0
        if o["kind"] == "polygon":
            xs = [p[0] for p in o["points"]]
            size = (max(xs) - min(xs)) or size
        self.outline_ = S.rectangle(size, size) if kind == "rectangle" else S.circle(size / 2) if kind == "circle" \
            else S.polygon([(0, 0), (size, 0), (size, size), (0, size)])
        self._fill()
        self.changed.emit()

    def _points(self):
        points = []
        for r in range(self.table.rowCount()):
            row = []
            for c in range(2):
                item = self.table.item(r, c)
                row.append(number(item.text()) if item is not None and item.text().strip() else None)
            if None in row:
                raise ValueError(f"point {r + 1} needs both x and y")
            points.append([float(row[0]), float(row[1])])
        return points

    def _edited(self):
        o = self.outline_
        if not self.fields.all_numbers():
            return
        try:
            if o["kind"] == "rectangle":
                self.outline_ = S.rectangle(*self.fields.values("width", "depth", "round"))
            elif o["kind"] == "circle":
                self.outline_ = S.circle(self.fields.value("radius"))
            else:
                self.outline_ = S.polygon(self._points(), self.fields.value("round"))
        except ValueError as e:
            self.error.setText(str(e))
            self.error.show()
            return
        self.error.hide()
        self.changed.emit()

    def _add_point(self):
        r = self.table.rowCount()
        self.table.blockSignals(True)
        self.table.insertRow(r)
        last = self.outline_["points"][-1] if self.outline_.get("points") else [0, 0]
        for c, v in enumerate(last):
            self.table.setItem(r, c, QtWidgets.QTableWidgetItem(format(v, ".12g")))
        self.table.blockSignals(False)
        self._edited()

    def _remove_point(self):
        if self.table.rowCount() > 3:
            self.table.removeRow(self.table.currentRow() if self.table.currentRow() >= 0 else self.table.rowCount() - 1)
            self._edited()

    def paste_points(self, text=None):
        """The table's points replaced by the clipboard's (or text): two numbers a row. A row that is not two
        numbers is named, and nothing changes."""
        text = QtWidgets.QApplication.clipboard().text() if text is None else text
        try:
            points = points_from_text(text)
        except ValueError as e:
            self.error.setText(f"Not pasted: {e}")
            self.error.show()
            return False
        self.outline_ = S.polygon(points, self.outline_.get("round", 0))
        self._fill()
        self.error.hide()
        self.changed.emit()
        return True

    def outline(self):
        return dict(self.outline_)


class ShapeEditor(QtWidgets.QWidget):
    """The boxes of one shape, by its kind; for a wrap, its placed shapes, one chosen at a time. changed() after
    an edit; shape() is the shape."""

    changed = QtCore.pyqtSignal()

    def __init__(self, shape, unit, size, parent=None):
        super().__init__(parent)
        self.shape_, self.unit, self.size = dict(shape), unit, size
        self.lay = QtWidgets.QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self._fill()

    def _fill(self):
        emptied(self.lay)
        s, u = self.shape_, self.unit
        self.fields, self.outline_editor, self.members = Fields(), None, None
        if s["kind"] == "box":
            self.fields.row("Size", ["x", "y", "z"], s["size"], u)
        elif s["kind"] == "cylinder":
            self.fields.row("Radius", ["radius"], [s["radius"]], u)
            self.fields.row("Height", ["height"], [s["height"]], u)
            if s.get("top_radius") is not None:
                self.fields.row("Top radius", ["top_radius"], [s["top_radius"]], u)
        elif s["kind"] == "sphere":
            self.fields.row("Radius", ["radius"], [s["radius"]], u)
        elif s["kind"] == "extrusion":
            self.fields.row("Height", ["height"], [s["height"]], u)
        elif s["kind"] == "revolution":
            self.fields.row("Angle", ["angle"], [s["angle"]], "degrees")
        self.fields.edited.connect(self._edited)
        if s["kind"] in ("extrusion", "revolution"):
            self.outline_editor = OutlineEditor(s["outline"], u, circle=s["kind"] == "extrusion")
            self.outline_editor.changed.connect(self._edited)
            self.lay.addWidget(self.outline_editor)
        self.lay.addWidget(self.fields)
        if s["kind"] == "cylinder":
            if s.get("top_radius") is None:
                self.lay.addWidget(link("+ Cone", lambda: self._cone(True)))
            else:
                self.lay.addWidget(link("Remove cone", lambda: self._cone(False)))
        if s["kind"] == "wrap":
            self._wrap_members()

    def _wrap_members(self):
        self.members = QtWidgets.QListWidget()
        self.members.setMaximumHeight(90)
        for k, m in enumerate(self.shape_["shapes"], 1):
            self.members.addItem(f"{k}  {m['shape']['kind']} at ({', '.join(format(v, 'g') for v in m['at'])})")
        self.lay.addWidget(QtWidgets.QLabel("Shapes wrapped"))
        self.lay.addWidget(self.members)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(link("+ Shape", self._add_member))
        row.addWidget(link("Remove shape", self._remove_member))
        row.addStretch()
        holder = QtWidgets.QWidget()
        holder.setLayout(row)
        self.lay.addWidget(holder)
        member_holder = QtWidgets.QWidget()                         # a widget, so _fill deletes it whole
        self.member_box = QtWidgets.QVBoxLayout(member_holder)
        self.member_box.setContentsMargins(0, 0, 0, 0)
        self.lay.addWidget(member_holder)
        self.members.currentRowChanged.connect(self._show_member)
        self.members.setCurrentRow(0)

    def _show_member(self, k):
        emptied(self.member_box)
        if k < 0:
            return
        m = self.shape_["shapes"][k]
        self.member_tiles = tiles(WRAPPED_TILES, lambda kind: self._member_kind(k, kind))
        self.member_tiles.hide()
        self.member_box.addWidget(link(f"{dict(TILES)[m['shape']['kind']]}: change",
                                       lambda: self.member_tiles.setVisible(not self.member_tiles.isVisible())))
        self.member_box.addWidget(self.member_tiles)
        self.member_editor = ShapeEditor(m["shape"], self.unit, self.size)
        self.member_place = Fields()
        self.member_place.row("At", ["x", "y", "z"], m["at"], self.unit)
        self.member_place.row("Turn", ["rx", "ry", "rz"], m["rotate"], "degrees")
        self.member_editor.changed.connect(lambda: self._member_edited(k))
        self.member_place.edited.connect(lambda: self._member_edited(k))
        self.member_box.addWidget(self.member_editor)
        self.member_box.addWidget(self.member_place)

    def _member_kind(self, k, kind):
        m = self.shape_["shapes"][k]
        if kind == m["shape"]["kind"]:
            self.member_tiles.hide()
            return
        self.shape_["shapes"][k] = S.place(default_shape(kind, self.size), m["at"], m["rotate"])
        self._refill_members(k)
        self.changed.emit()

    def _member_edited(self, k):
        if not self.member_place.all_numbers():
            return
        self.shape_["shapes"][k] = S.place(self.member_editor.shape(), self.member_place.values("x", "y", "z"),
                                           self.member_place.values("rx", "ry", "rz"))
        self.members.item(k).setText(f"{k + 1}  {self.shape_['shapes'][k]['shape']['kind']} at "
                                     f"({', '.join(format(v, 'g') for v in self.shape_['shapes'][k]['at'])})")
        self.changed.emit()

    def _add_member(self):
        self.shape_["shapes"].append(S.place(S.cylinder(self.size / 4, self.size / 4), at=(self.size, self.size, 0)))
        self._refill_members(len(self.shape_["shapes"]) - 1)
        self.changed.emit()

    def _remove_member(self):
        k = self.members.currentRow()
        if k >= 0 and len(self.shape_["shapes"]) > 2:              # a wrap is of two shapes or more
            del self.shape_["shapes"][k]
            self._refill_members(max(0, k - 1))
            self.changed.emit()

    def _refill_members(self, k):
        self._fill()
        self.members.setCurrentRow(k)

    def _cone(self, on):
        s = self.shape_
        self.shape_ = S.cylinder(s["radius"], s["height"], s["radius"] / 2 if on else None)
        self._fill()
        self.changed.emit()

    def _edited(self):
        s, f = self.shape_, self.fields
        if not f.all_numbers():
            return
        if s["kind"] == "box":
            self.shape_ = S.box(*f.values("x", "y", "z"))
        elif s["kind"] == "cylinder":
            top = f.value("top_radius") if "top_radius" in f.boxes else None
            self.shape_ = S.cylinder(f.value("radius"), f.value("height"), top)
        elif s["kind"] == "sphere":
            self.shape_ = S.sphere(f.value("radius"))
        elif s["kind"] == "extrusion":
            self.shape_ = S.extrusion(self.outline_editor.outline(), f.value("height"))
        elif s["kind"] == "revolution":
            self.shape_ = S.revolution(self.outline_editor.outline(), f.value("angle"))
        self.changed.emit()

    def shape(self):
        return dict(self.shape_)


class StepEditor(QtWidgets.QWidget):
    """The boxes of one step: what it does (a switch), its shape (its kind, which a click on it changes, and its
    boxes), At, and Turn and Repeat, open only when asked for or in use. changed() after an edit; step() is
    the step."""

    changed = QtCore.pyqtSignal()

    def __init__(self, step, unit, size, first, parent=None):
        super().__init__(parent)
        self.step_, self.unit, self.size, self.first = dict(step), unit, size, first
        self.turn_open = any(step["rotate"])
        self.lay = QtWidgets.QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self._fill()

    def _fill(self):
        emptied(self.lay)
        st, u = self.step_, self.unit
        top = QtWidgets.QHBoxLayout()
        self.action = Switch([("add", "Add"), ("cut", "Cut"), ("overlap", "Overlap")], st["action"])
        self.action.setEnabled(not self.first)
        self.action.setToolTip("The first step adds a shape" if self.first else "")
        self.action.changed.connect(self._edited)
        top.addWidget(self.action, stretch=1)
        kind = dict(TILES)[st["shape"]["kind"]]
        top.addWidget(link(f"{kind}: change", self._show_tiles))
        holder = QtWidgets.QWidget()
        holder.setLayout(top)
        self.lay.addWidget(holder)
        self.tile_box = tiles(TILES, self._kind_chosen)
        self.tile_box.hide()
        self.lay.addWidget(self.tile_box)
        self.shape_editor = ShapeEditor(st["shape"], u, self.size)
        self.shape_editor.changed.connect(self._edited)
        self.lay.addWidget(self.shape_editor)
        self.place = Fields()
        self.place.row("At", ["x", "y", "z"], st["at"], u)
        if self.turn_open:
            self.place.row("Turn", ["rx", "ry", "rz"], st["rotate"], "degrees")
        self.place.edited.connect(self._edited)
        self.lay.addWidget(self.place)
        if not self.turn_open:
            self.lay.addWidget(link("+ Turn", self._open_turn))
        self._repeat_part()

    def _repeat_part(self):
        r = self.step_.get("repeat")
        self.repeat_kind, self.repeat_fields = None, None
        if r is None:
            self.lay.addWidget(link("+ Repeat", lambda: self._set_repeat("along_line")))
            return
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Repeat"))
        self.repeat_kind = Switch([("along_line", "Along a line"), ("around_circle", "Around a circle")], r["kind"])
        self.repeat_kind.changed.connect(self._set_repeat)
        row.addWidget(self.repeat_kind, stretch=1)
        holder = QtWidgets.QWidget()
        holder.setLayout(row)
        self.lay.addWidget(holder)
        self.repeat_fields = Fields()
        self.repeat_fields.row("Count", ["count"], [r["count"]], integer=True)
        if r["kind"] == "along_line":
            self.repeat_fields.row("Spacing", ["dx", "dy", "dz"], r["step"], self.unit)
        else:
            self.repeat_fields.row("Centre", ["cx", "cy", "cz"], r["centre"], self.unit)
        self.repeat_fields.edited.connect(self._edited)
        self.lay.addWidget(self.repeat_fields)
        if r["kind"] == "around_circle":
            axis = QtWidgets.QHBoxLayout()
            axis.addWidget(QtWidgets.QLabel("Axis"))
            self.axis = Switch([("x", "x"), ("y", "y"), ("z", "z")], r["axis"])
            self.axis.changed.connect(self._edited)
            axis.addWidget(self.axis, stretch=1)
            holder = QtWidgets.QWidget()
            holder.setLayout(axis)
            self.lay.addWidget(holder)
        self.lay.addWidget(link("Remove repeat", lambda: self._set_repeat(None)))

    def _open_turn(self):
        self.turn_open = True
        self._fill()

    def _show_tiles(self):
        self.tile_box.setVisible(not self.tile_box.isVisible())

    def _kind_chosen(self, kind):
        st = self.step_
        if kind != st["shape"]["kind"]:
            st["shape"] = default_shape(kind, self.size)
        self._fill()
        self.changed.emit()

    def _set_repeat(self, kind):
        r = self.step_.get("repeat")
        count = r["count"] if r else 2
        self.step_["repeat"] = None if kind is None else S.along_line(count, (self.size, 0, 0)) \
            if kind == "along_line" else S.around_circle(count, (0, 0, 0), "z")
        self._fill()
        self.changed.emit()

    def _edited(self, *_):
        st = self.step_
        boxes = [self.place] + ([self.repeat_fields] if self.repeat_fields is not None else [])
        if not all(f.all_numbers() for f in boxes):
            return
        st["action"] = "add" if self.first else self.action.value()
        st["shape"] = self.shape_editor.shape()
        st["at"] = [float(v) for v in self.place.values("x", "y", "z")]
        st["rotate"] = [float(v) for v in self.place.values("rx", "ry", "rz")] if self.turn_open else [0.0] * 3
        if self.repeat_fields is not None:
            count = self.repeat_fields.value("count")
            if self.repeat_kind.value() == "along_line":
                st["repeat"] = S.along_line(count, self.repeat_fields.values("dx", "dy", "dz"))
            else:
                st["repeat"] = S.around_circle(count, self.repeat_fields.values("cx", "cy", "cz"), self.axis.value())
        self.changed.emit()

    def step(self):
        return dict(self.step_)


class GeometryPanel(QtWidgets.QDialog):
    """The Geometry strip: the project's part from an STL file (Browse..., Scale...) or from shapes (the steps,
    the chosen step's boxes, Update part, Scale...). Edits go into the project's steps at once; the part is
    built and drawn on Update part, and on closing the strip."""

    def __init__(self, window):
        super().__init__(window)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setWindowTitle("Geometry")
        self.setMinimumWidth(340)
        self.window = window
        self.adding = False                                         # the tiles of a new step are shown
        outer = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.side = Switch([("stl", "From an STL file"), ("shapes", "From shapes")],
                           "shapes" if window.project.part_shapes is not None else "stl")
        self.side.changed.connect(self._side_changed)
        top.addWidget(self.side, stretch=1)
        top.addWidget(Info(GEOMETRY_HELP))
        outer.addLayout(top)
        self.pages = QtWidgets.QStackedWidget()
        outer.addWidget(self.pages, stretch=1)
        self.pages.addWidget(self._stl_page())
        self.pages.addWidget(self._shapes_page())
        self.message = QtWidgets.QLabel("")
        self.message.setWordWrap(True)
        outer.addWidget(self.message)
        bottom = QtWidgets.QHBoxLayout()
        self.update_button = QtWidgets.QPushButton("Update part")
        self.update_button.setObjectName("updatePart")
        self.update_button.clicked.connect(lambda _=False: self.update_part())
        bottom.addWidget(self.update_button)
        self.scale_button = QtWidgets.QPushButton("Scale...")
        self.scale_button.clicked.connect(lambda _=False: self._scale())
        bottom.addWidget(self.scale_button)
        bottom.addStretch()
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.close)
        bottom.addWidget(close)
        outer.addLayout(bottom)
        self._side_changed(self.side.value())

    # ------------------------------------------------------------------ the two sides
    def _stl_page(self):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 8, 0, 0)
        self.stl_name = QtWidgets.QLabel("")
        self.stl_name.setWordWrap(True)
        lay.addWidget(self.stl_name)
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(lambda _=False: self._browse())
        row = QtWidgets.QHBoxLayout()
        row.addWidget(browse)
        row.addStretch()
        lay.addLayout(row)
        lay.addStretch()
        return page

    def _shapes_page(self):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 8, 0, 0)
        self.steps_list = QtWidgets.QListWidget()
        self.steps_list.setMaximumHeight(150)
        self.steps_list.currentRowChanged.connect(self._step_chosen)
        lay.addWidget(self.steps_list)
        row = QtWidgets.QHBoxLayout()
        self.step_buttons = {}
        for text, slot in (("New step", self.new_step), ("Up", lambda: self.move_step(-1)),
                           ("Down", lambda: self.move_step(1)), ("Copy", self.copy_step), ("Delete", self.delete_step)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _=False, s=slot: s())
            row.addWidget(b)
            self.step_buttons[text] = b
        row.addStretch()
        lay.addLayout(row)
        self.editor_area = QtWidgets.QScrollArea()
        self.editor_area.setWidgetResizable(True)
        self.editor_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        lay.addWidget(self.editor_area, stretch=1)
        return page

    def _side_changed(self, side):
        self.pages.setCurrentIndex(0 if side == "stl" else 1)
        self.update_button.setVisible(side == "shapes")
        p = self.window.project
        self.stl_name.setText(f"Part: {os.path.basename(p.stl)}" if p.stl is not None else "No STL file chosen yet.")
        if side == "shapes":
            self._fill_list()
        else:
            self.window.show_step_shape(None)
        self._show_message()

    # ------------------------------------------------------------------ the steps
    def steps(self):
        return list(self.window.project.part_shapes or [])

    def size(self):
        """About a fifth of the part's size, rounded, for a shape to start from; 10 with no part drawn."""
        b = self.window.part_bounds()
        return nice(float(max(b[1] - b[0])) / 5) if b is not None else 10.0

    def _fill_list(self, chosen=None):
        steps = self.steps()
        self.steps_list.blockSignals(True)
        self.steps_list.clear()
        for step in steps:
            self.steps_list.addItem(line_of(step))
        self.steps_list.blockSignals(False)
        if steps:
            self.steps_list.setCurrentRow(len(steps) - 1 if chosen is None else chosen)
        else:
            self.new_step()

    def _set_steps(self, steps, chosen=None):
        """The project's steps made these; the window waits for Update part."""
        self.window.project.part_shapes = steps
        self.window.part_steps_changed()
        self._fill_list(chosen)
        self._show_message()

    def _step_chosen(self, k):
        self.adding = False
        steps = self.steps()
        if not 0 <= k < len(steps):
            return
        editor = StepEditor(steps[k], self.window.length_unit(), self.size(), first=k == 0)
        editor.changed.connect(lambda: self._step_edited(k, editor))
        self.editor_area.setWidget(editor)
        self._highlight(steps[k])

    def _step_edited(self, k, editor):
        steps = self.steps()
        steps[k] = editor.step()
        self.window.project.part_shapes = steps
        self.window.part_steps_changed()
        self.steps_list.item(k).setText(line_of(steps[k]))
        self._highlight(steps[k])
        self._show_message()

    def _highlight(self, step):
        """The step's shape drawn see-through over the part, so a cut that misses shows where it is."""
        from lsr_mmto.problem.part_from_shapes import build_surface
        try:
            surface, _ = build_surface([dict(step, action="add")])
        except (ShapeError, ValueError):
            surface = None
        self.window.show_step_shape(surface)

    def new_step(self):
        """The tiles of a new step, under the list; a click on one adds the step."""
        self.adding = True
        self.steps_list.clearSelection()
        box = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        steps = self.steps()
        if steps:
            self.new_action = Switch([("add", "Add"), ("cut", "Cut"), ("overlap", "Overlap")], "add")
            lay.addWidget(self.new_action)
        else:
            self.new_action = None
            lay.addWidget(QtWidgets.QLabel("The first step adds a shape. Choose it:"))
        lay.addWidget(tiles(TILES, self.add_step))
        lay.addStretch()
        self.editor_area.setWidget(box)
        self.window.show_step_shape(None)
        return True

    def add_step(self, kind):
        """A step of this kind of shape added at the end, from where the shapes start (default_shape). A part that
        is an STL file is made from shapes only when the user agrees; None when not."""
        from .main_window import triangle_selections
        w = self.window
        if w.template:                                              # an example is never written: a copy first
            answer = QtWidgets.QMessageBox.question(
                self, "Geometry", "This is an example, which stays as it is. Save it as a project of your own "
                                  "first?", QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Cancel)
            if answer != QtWidgets.QMessageBox.Save or not w.save_project() or w.template:
                return None
        p = w.project
        if p.stl is not None:
            picked = triangle_selections(p)
            note = f" These selections are faces picked on it, to pick again afterwards: {', '.join(picked)}." \
                if picked else ""
            answer = QtWidgets.QMessageBox.question(
                self, "Geometry", f"The part is the STL file {os.path.basename(p.stl)}. Make it from shapes instead? "
                                  f"The STL is no longer used.{note}")
            if answer != QtWidgets.QMessageBox.Yes:
                return None
            p.stl = None
        action = self.new_action.value() if self.new_action is not None else "add"
        step = {"add": S.add, "cut": S.cut, "overlap": S.overlap}[action](default_shape(kind, self.size()))
        steps = self.steps() + [step]
        self._set_steps(steps, len(steps) - 1)
        return step

    def _first_adds(self, steps):
        """Whether the first step adds a shape, as it must; when not, said in red and nothing changes."""
        if steps and steps[0]["action"] != "add":
            self._show_message("the first step must add a shape, so this would leave a Cut or an Overlap first")
            return False
        return True

    def move_step(self, by):
        k, steps = self.steps_list.currentRow(), self.steps()
        if 0 <= k < len(steps) and 0 <= k + by < len(steps):
            steps[k], steps[k + by] = steps[k + by], steps[k]
            if self._first_adds(steps):
                self._set_steps(steps, k + by)

    def copy_step(self):
        k, steps = self.steps_list.currentRow(), self.steps()
        if 0 <= k < len(steps):
            steps.insert(k + 1, S.plain([steps[k]])[0])
            self._set_steps(steps, k + 1)

    def delete_step(self):
        k, steps = self.steps_list.currentRow(), self.steps()
        if 0 <= k < len(steps):
            del steps[k]
            if self._first_adds(steps):
                self._set_steps(steps, min(k, len(steps) - 1) if steps else None)

    # ------------------------------------------------------------------ the part
    def _show_message(self, error=None):
        """Under the steps: a part that cannot be made (red), or the warnings; and that the view waits."""
        lines, style = [], ""
        if error:
            lines, style = [f"Not done: {error}" if error.startswith("the first step must") else
                            f"The part cannot be made: {error}"], RED
        elif self.side.value() == "shapes" and not self.window.part_waiting and self.window.project.has_part():
            try:
                lines = self.window.project.part_warnings()
            except (ShapeError, OSError):
                lines = []
        if self.window.part_waiting and self.side.value() == "shapes":
            lines.append(WAITING)
        self.message.setText("\n".join(lines))
        set_style(self.message, style or GREY)
        self.scale_button.setEnabled(self.window.project.has_part())
        self.scale_button.setToolTip("" if self.window.project.has_part() else "Add a part first")
        self.update_button.setProperty("waiting", "true" if self.window.part_waiting else "false")
        self.update_button.style().unpolish(self.update_button)
        self.update_button.style().polish(self.update_button)

    def update_part(self):
        """The part built from the steps and drawn, alone: the chosen step's shape shows again once a step is
        chosen or changed. A part that cannot be made is said here in red, the view keeps the last part built
        and the chosen step's shape stays. True when built."""
        if not self.window.project.part_shapes:
            self.window.update_part_from_shapes()                   # no part: the drawing goes, the checks follow
            self._show_message("there are no steps yet")
            return False
        error = self.window.update_part_from_shapes()
        k = self.steps_list.currentRow()
        if error is None:
            self.window.show_step_shape(None)
        elif 0 <= k < len(self.steps()):
            self._highlight(self.steps()[k])
        self._show_message(error)
        return error is None

    def _browse(self):
        if self.window.replace_part():
            self.stl_name.setText(f"Part: {os.path.basename(self.window.project.stl)}")
            self.steps_list.clear()
            self._show_message()

    def _scale(self):
        if self.window.part_waiting and not self.update_part():
            return                                                  # a part that cannot be made: not scaled
        if self.window.scale_part():
            if self.side.value() == "shapes":
                self._fill_list(self.steps_list.currentRow())
            self._show_message()

    def closeEvent(self, event):
        if self.window.part_waiting and self.window.project is not None:
            self.update_part()                                      # never left with the view behind the steps
        self.window.show_step_shape(None)
        super().closeEvent(event)
