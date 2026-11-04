"""
supports_and_loads_panel.py
===========================
The Supports and loads panel: the project's regions (selections), its supports and loads,
and which regions are kept solid, with a preview on the mesh the run will use.

A region is made three ways: picked with the mouse in PyTO-AD's Structural Loads window (each
Apply is adopted into the project at once, by Project.adopt_picks), filled in on a form made
from the selection type's own fields, or typed as a code line, e.g.
rows(face="xmax", axis="y", n=8, end="max"). A code line is read, never run: only selection
types with plain numbers, text and lists are accepted; a number may be plain arithmetic (38/2, 2*pi*5).

The panel works on the project directly (it stays open while you pick); File > Save keeps the
changes, and closing without saving drops them.
"""
import ast
import inspect
import os

import numpy as np
from PyQt5 import QtCore, QtWidgets

from .panel_widgets import GREY, Info, NumberEdit, plain_value

SUPPORT_COLOUR, LOAD_COLOUR, KEEP_COLOUR, OTHER_COLOUR = "#1f5fd6", "#d62020", "#808080", "#e08a00"

# What the region form shows for each selection type: its name, what it is, and a label per field.
# The code names (the selection types and their fields) stay on the Code line tab.
FORM = {
    "face": ("Face of the part", "All nodes on one outer face of the part.", {"side": "Side"}),
    "rows": ("Rows on a face", "A few rows of nodes at one end of a face. For example, the top 8 rows of the "
             "right face: Face Right, Counted along y, Number of rows 8, From High end. Use it to spread a load "
             "over a small patch.",
             {"face": "Face", "axis": "Counted along", "n": "Number of rows", "end": "From"}),
    "near_face": ("Near a face", "Nodes within a set distance of one outer face.",
                  {"side": "Side", "distance": "Distance"}),
    "plane": ("Flat cut", "Nodes on a flat cut through the part. For example, every node where y is 38: "
              "Cut across y, At 38. To keep only part of the cut, fill in the from and to of the other "
              "directions, e.g. x from 0 to 40; leave them empty for the whole cut.",
              {"axis": "Cut across", "value": "At", "limits": "from / to"}),
    "box": ("Box", "Nodes inside a box given by two opposite corners.", {"lo": "Corner 1", "hi": "Corner 2"}),
    "sphere": ("Sphere", "Nodes within a radius of a point.", {"center": "Centre", "radius": "Radius"}),
    "cylinder": ("Cylinder", "Nodes inside a cylinder, or a tube if you give an inner radius. Good for holes "
                 "and bolt bores. Leave the ends empty for a cylinder through the whole part. With the axis "
                 "along x, y or z the ends are coordinates, e.g. x from 22 to 61; along a tilted axis they "
                 "are distances from the point on the axis.",
                 {"center": "Point on the axis", "axis": "Axis direction", "radius": "Radius",
                  "inner_radius": "Inner radius (0 = solid)", "along": ("From", "To")}),
    "nearest": ("Closest node", "The one node closest to a point.",
                {"point": "Point", "boundary": "Surface nodes only"}),
    "inside_stl": ("Inside an STL", "Nodes inside a closed surface loaded from an STL file. Move by moves the "
                   "shape first.", {"path": "File", "shift": "Move by"}),
    "combined": ("Combined selections", "Two selections of this project joined, overlapped or one taken from "
                 "the other. It follows them: edit Selection A or Selection B later and this selection changes "
                 "with it.",
                 {"how": "How", "a": "Selection A", "b": "Selection B"}),
    # made with the mouse: offered only when editing a region that is one
    "stl_faces": ("Picked faces", "Faces you clicked with the mouse. Snap distance is how close a node must "
                  "be to a face to count, in element sizes; leave it empty for the usual 0.7.",
                  {"triangles": "Face numbers", "tol": "Snap distance"}),
    "halfspaces": ("Dragged box", "A box you dragged with the mouse, as the planes that bound it.",
                   {"normals": "Plane directions", "origins": "Points on the planes",
                    "boundary": "Surface nodes only"}),
    "ids": ("Node numbers", "Specific node numbers. They stay right only while the mesh does not change.",
            {"indices": "Node numbers"}),
}
PICKED = {"stl_faces", "halfspaces", "ids"}
HOW = [("Join: in A or in B", "union"), ("Overlap: in both A and B", "intersect"),
       ("A minus B: in A, not in B", "difference")]


def combined_parts(selection):
    """(how, A, B) of a union, intersect or difference of two named selections, or None."""
    from lsr_mmto.problem import selections as S
    d = selection.to_dict()
    parts = d.get("of", [])
    if d["type"] in dict((v, k) for k, v in HOW) and len(parts) == 2 and all(p["type"] in S.LINKS for p in parts):
        return d["type"], parts[0]["name"], parts[1]["name"]
    return None
TOLERANCE = ("Tolerance", "How far outside the selection a node may sit and still count. Automatic is right "
                          "almost always; or type a length.")
PLACES_HELP = ("A selection is the part of the model a support or load acts on: faces you clicked, or a "
               "selection you described (a face of the part, a box, a cylinder...). Each selection has a name, "
               "like clamp or corner, which shows under Where, so several supports and loads can share one "
               "selection. Open Selections to rename a selection, change it, or tick Keep solid so the "
               "optimization never removes it.")
MOUSE_HINT = "Hover over the part to read its coordinates; click it to copy the point."


def field_kind(cls, field):
    """How a field of a selection type is given (its Field: a length, a point, a choice...), as the
    type itself declares it; a plain text box for one it does not declare."""
    from lsr_mmto.problem.selections import Field
    return getattr(cls, "field_kinds", {}).get(field) or Field("text")


# ---------------------------------------------------------------------- regions as text
def selection_types():
    """{type name: class} of the selection types the GUI offers: all but function, which runs a
    Python file when it is evaluated (scripts can still use it)."""
    from lsr_mmto.problem import selections as S
    return {k: v for k, v in sorted(S._TYPES.items()) if k != "function"}


def parse_selection(text):
    """A Selection from a code line such as box(lo=[0, 0, 0], hi=[1, 2, 1]). Only selection types,
    with numbers (plain arithmetic too: 38/2), text, lists and other selections as arguments; anything
    else is refused."""
    types = selection_types()
    try:
        tree = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as e:
        raise ValueError(f"not a selection: {e.msg}") from None

    def build(node):
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in types:
                name = getattr(node.func, "id", ast.unparse(node.func))
                raise ValueError(f"{name} is not a selection type; known: {', '.join(types)}")
            if any(k.arg is None for k in node.keywords):
                raise ValueError("** arguments are not accepted")
            args = [build(a) for a in node.args]
            kwargs = {k.arg: build(k.value) for k in node.keywords}
            try:
                return types[node.func.id](*args, **kwargs)
            except TypeError as e:
                raise ValueError(f"{node.func.id}: {e}") from None
        try:
            return plain_value(node)
        except ValueError:
            raise ValueError(f"{ast.unparse(node)!r}: only numbers, text, lists and selections are "
                             f"accepted") from None

    selection = build(tree)
    if not hasattr(selection, "evaluate"):
        raise ValueError("write a selection, e.g. face(\"xmin\")")
    return selection


def shown_name(kind):
    """The region type as the form shows it."""
    return FORM[kind][0] if kind in FORM else kind


def field_label(kind, field):
    if field == "pad":
        return TOLERANCE[0]
    label = FORM.get(kind, ("", "", {}))[2].get(field, field)
    return " / ".join(label) if isinstance(label, tuple) else label


def show_value(value, choices=()):
    """A field's value as the form and the tables show it: a choice by its name (a side as Right (x max)),
    points without brackets."""
    for shown, v in choices:
        if value == v:
            return shown
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".12g")                                # 5, not 5.0; 12 digits: kept exactly
    if isinstance(value, (list, tuple)) and all(isinstance(x, (int, float)) for x in value):
        return ", ".join(format(x, ".12g") for x in value)              # 0, -4, 0; 12 digits: kept exactly
    return value if isinstance(value, str) else repr(value)


def describe(selection):
    """A region in words, for the regions table: e.g. Face of the part: Side Right (x max)."""
    kind = selection.TYPE
    combo = combined_parts(selection)
    if combo:
        how = {"union": "join of", "intersect": "overlap of", "difference": "minus"}[combo[0]]
        return (f"Combined selections: {combo[1]} minus {combo[2]}" if combo[0] == "difference" else
                f"Combined selections: {how} {combo[1]} and {combo[2]}")
    if kind not in FORM:
        return repr(selection)
    defaults = dict(type_fields(type(selection)))
    parts = []
    for field in selection.fields:
        value = getattr(selection, field)
        if value is None or value == defaults.get(field) or show_value(value) == show_value(defaults.get(field)):
            continue                                                # as by default (0, 0, 0 as a list too)
        if field == "along":
            axis = getattr(selection, "axis", None)
            if isinstance(axis, str) and axis in "xyz" and all(isinstance(v, (int, float)) for v in value):
                c = selection.center["xyz".index(axis)]
                parts.append(f"{axis} from {show_value(c + value[0])} to {show_value(c + value[1])}")
            else:
                parts.append(f"from {show_value(value[0])} to {show_value(value[1])} along the axis")
        else:
            parts.append(f"{field_label(kind, field)} {show_value(value, field_kind(type(selection), field).choices)}")
    return f"{shown_name(kind)}: " + ", ".join(parts)


def type_fields(cls):
    """[(field, default or inspect.Parameter.empty)] of a selection type, from its __init__."""
    params = inspect.signature(cls.__init__).parameters
    return [(f, params[f].default if f in params else inspect.Parameter.empty) for f in cls.fields]


def used_by(project, name):
    """What uses the region: its supports and loads, and keep solid."""
    from lsr_mmto.problem import selections as S
    out = [f"{bc_name(bc['kind'])} {i}" for i, bc in enumerate(project.bcs, 1) if bc.get("on") == name]
    out += [f"combined selection {k}" for k, s in project.selections.items() if name in S.names_in(s)]
    return out + (["keep solid"] if name in project.keep_solid_names else [])


def enter_previews(buttons, preview):
    """Enter in a form previews, it does not press OK: the Preview button is the default one, and no
    button is pressed by Enter just because it has the focus."""
    for button in buttons.buttons():
        button.setAutoDefault(False)
    preview.setDefault(True)


def preview_on_enter(dialog, event):
    """Enter or Return in the form: dialog.show_preview() and True; the window stays open."""
    if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
        dialog.show_preview()
        return True
    return False


NOT_ROUND = "This doesn't look like a round hole"


def hole_from_wall(normals, corners):
    """(point on the axis, axis, radius, start, end) of a round hole, from the triangles of its inside wall
    (their normals, and their corners as an (n, 3, 3) array). The axis is the direction the wall's normals
    avoid: the eigenvector of the smallest eigenvalue of the sum of n n^T, exactly x, y or z when within
    0.01 % of one. The centre and radius are the least-squares circle (algebraic fit) through the corners
    seen along the axis; start and end are the wall's extent along the axis, and the point is on the axis
    at start. ValueError when the corners are far from a circle: rms off it over 5 % of the radius, or all
    on one line (no circle to fit)."""
    n = np.asarray(normals, dtype=float)
    n = n / np.linalg.norm(n, axis=1)[:, None]
    p = np.unique(np.asarray(corners, dtype=float).reshape(-1, 3), axis=0)
    axis = np.linalg.eigh(n.T @ n)[1][:, 0]
    k = int(np.argmax(np.abs(axis)))
    axis = axis * np.sign(axis[k])
    if abs(axis[k]) > 0.9999:
        axis = np.eye(3)[k]
    u = np.cross(axis, np.eye(3)[(k + 1) % 3])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    middle = p.mean(axis=0)                                         # the fit about the corners' middle
    x, y, t = (p - middle) @ u, (p - middle) @ v, p @ axis
    fit = np.column_stack([x, y, np.ones_like(x)])
    (d, e, f), _, _, s = np.linalg.lstsq(fit, -(x * x + y * y), rcond=None)
    cx, cy = -d / 2, -e / 2
    r2 = cx * cx + cy * cy - f
    if s[-1] < 1e-6 * s[0] or r2 <= 0:
        raise ValueError(NOT_ROUND)
    r = float(np.sqrt(r2))
    if np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)) > 0.05 * r:
        raise ValueError(NOT_ROUND)
    centre = middle - (middle @ axis) * axis + cx * u + cy * v + t.min() * axis
    return centre, axis, r, float(t.min()), float(t.max())


def _number_text(value):
    return format(round(float(value), 9) + 0.0, ".6g")


def fill_from_hole(dialog, normals, corners):
    """The Cylinder form filled in from the inside wall of a hole (hole_from_wall): Point on the axis, Axis
    direction, Radius (the hole's radius + 1h, one element size of whatever mesh it is used on, so the nodes
    of the wall are in), and the ends; then its preview. A wall that is not round fills nothing and says so."""
    try:
        centre, axis, r, start, end = hole_from_wall(normals, corners)
    except ValueError as e:
        dialog.result.setText(f"<span style='color:#b00000'>{e}</span>")
        return
    letter = next((a for a, e in zip("xyz", np.eye(3)) if np.array_equal(axis, e)), None)
    boxes = dialog.inputs
    boxes["center"].setText(", ".join(_number_text(c) for c in centre))
    boxes["axis"].setCurrentText(letter or ", ".join(_number_text(c) for c in axis))
    boxes["radius"].setText(f"{_number_text(r)} + 1h")
    ends = (start, end) if letter else (0.0, end - start)          # coordinates along x, y, z; else distances
    for box, value in zip(boxes["along"], ends):
        box.setText(_number_text(value))
    dialog.show_preview()


# A button of the form that fills it in from a click on the part: (its text, its (i), the filling, given the
# dialog and the normals and corners of the face clicked).
CLICK_TO_FILL = {"cylinder": ("Click a hole", "Then click the inside wall of a round hole on the part: the "
                              "point, axis, radius and ends are filled in from it.", fill_from_hole)}


# ---------------------------------------------------------------------- a region
class RegionDialog(QtWidgets.QDialog):
    """A region: its name, and its definition on a form or as a code line."""

    def __init__(self, parent, project, name=None, preview=None):
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.setWindowTitle("Selection")
        self.resize(520, 380)
        self.project, self.old_name, self.preview = project, name, preview
        layout = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Name"))
        self.name = QtWidgets.QLineEdit(name or project._free_name("selection"))
        row.addWidget(self.name)
        layout.addLayout(row)

        self.tabs = QtWidgets.QTabWidget()
        form_page = QtWidgets.QWidget()
        form_layout = QtWidgets.QVBoxLayout(form_page)
        self.type = QtWidgets.QComboBox()
        current = project.selections.get(name)
        kept = {current.TYPE} if current is not None else set()      # a picked region: its own type too
        self.types = {k: c for k, c in selection_types().items() if k in FORM and (k not in PICKED or k in kept)}
        self.others = [n for n in project.selections if n != name]  # selections this one may combine
        if len(self.others) >= 2:
            self.types["combined"] = None
        for k in sorted(self.types, key=list(FORM).index):
            self.type.addItem(shown_name(k), k)
        self.type_info = Info("")
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.type, stretch=1)
        top.addWidget(self.type_info)
        form_layout.addLayout(top)
        main = parent                                               # the main window: the part's extent
        while main is not None and not hasattr(main, "part_extent_text"):
            main = main.parentWidget()
        self.main = main
        self.on_click = None                                        # a click on the part it waits for
        extent = QtWidgets.QLabel(main.part_extent_text() if main is not None else "")
        extent.setStyleSheet(GREY)
        extent.setVisible(bool(extent.text()))
        form_layout.addWidget(extent)
        self.form = QtWidgets.QFormLayout()
        form_layout.addLayout(self.form)
        form_layout.addStretch()
        self.length_unit = field_unit("length", project.units)
        self.tabs.addTab(form_page, "Form")
        code_page = QtWidgets.QWidget()
        code_layout = QtWidgets.QVBoxLayout(code_page)
        self.code = QtWidgets.QLineEdit()
        code_layout.addWidget(self.code)
        hint = QtWidgets.QLabel("For example rows(face=\"xmax\", axis=\"y\", n=8, end=\"max\"), or selections "
                                "combined: union(face(\"xmin\"), box(lo=[0, 0, 0], hi=[2, 10, 1]))")
        hint.setStyleSheet(GREY)
        hint.setWordWrap(True)
        code_layout.addWidget(hint)
        code_layout.addStretch()
        self.tabs.addTab(code_page, "Code line")
        layout.addWidget(self.tabs)

        self.result = QtWidgets.QLabel()
        self.result.setWordWrap(True)
        layout.addWidget(self.result)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        show = buttons.addButton("Preview", QtWidgets.QDialogButtonBox.ActionRole)
        show.setToolTip("Check what you typed and show it on the part (Enter)")
        show.clicked.connect(self.show_preview)
        clear = buttons.addButton("Clear", QtWidgets.QDialogButtonBox.ActionRole)
        clear.setToolTip("Clear this preview from the part")
        clear.clicked.connect(self.clear_shown)
        self.shown = False                                          # a preview of this form is on the part
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        enter_previews(buttons, show)
        layout.addWidget(buttons)

        self.inputs = {}
        self.type.currentIndexChanged.connect(lambda _: (self.changed(), self._fill_form(self.kind())))
        current_kind = None if current is None else "combined" if combined_parts(current) else current.TYPE
        if current is not None and current_kind in self.types:
            self.type.blockSignals(True)
            self.type.setCurrentIndex(self.type.findData(current_kind))
            self.type.blockSignals(False)
            self._fill_form(current_kind, current)
        else:
            self._fill_form(self.kind())
        if current is not None:
            self.code.setText(repr(current))
            if current_kind not in self.types:
                self.tabs.setCurrentIndex(1)
        self.tabs.currentChanged.connect(self._switch)              # after: opening on the code line copies nothing
        self.code.textChanged.connect(self.changed)

    def kind(self):
        return self.type.currentData()

    def _fill_form(self, kind, selection=None):
        while self.form.rowCount():
            self.form.removeRow(0)
        cls = self.types[kind]
        self.type_info.setToolTip(FORM[kind][1])
        self.inputs = {}
        self.on_click = None
        if kind == "combined":                                      # How, Selection A, Selection B: by name, live
            how, a, b = combined_parts(selection) if selection is not None else ("union", None, None)
            boxes = {"how": QtWidgets.QComboBox(), "a": QtWidgets.QComboBox(), "b": QtWidgets.QComboBox()}
            for shown, v in HOW:
                boxes["how"].addItem(shown, v)
            boxes["how"].setCurrentIndex(boxes["how"].findData(how))
            for key, pick, default in (("a", a, 0), ("b", b, 1)):
                boxes[key].addItems(self.others)
                boxes[key].setCurrentIndex(self.others.index(pick) if pick in self.others else default)
            for key, box in boxes.items():
                self.form.addRow(FORM["combined"][2][key], box)
                box.currentIndexChanged.connect(self.changed)
            self.inputs = boxes
            return
        for field, default in type_fields(cls):
            value = getattr(selection, field) if selection is not None else \
                None if default is inspect.Parameter.empty else default
            if field == "limits":                                   # from / to for the other two axes
                box = {}
                for a in "xyz":
                    ends = (value or {}).get(a, ("", ""))
                    pair = tuple(QtWidgets.QLineEdit("" if e == "" else show_value(e)) for e in ends)
                    row = QtWidgets.QWidget()
                    row_layout = QtWidgets.QHBoxLayout(row)
                    row_layout.setContentsMargins(0, 0, 0, 0)
                    for b in pair:
                        b.setPlaceholderText("optional")
                        b.textChanged.connect(self.changed)
                        row_layout.addWidget(b)
                    if self.length_unit:
                        unit = QtWidgets.QLabel(self.length_unit)
                        unit.setStyleSheet(GREY)
                        row_layout.addWidget(unit)
                    self.form.addRow(f"{a} from / to", row)
                    box[a] = (pair, row)
                self.inputs[field] = box
                self.inputs["axis"].currentIndexChanged.connect(self._show_limits)
                self._show_limits()
                continue
            if field == "along":                                    # two boxes: the ends
                ends = list(value) if value else ["", ""]
                axis = getattr(selection, "axis", None)
                if value and isinstance(axis, str) and axis in "xyz" and \
                        all(isinstance(e, (int, float)) for e in ends):     # shown as coordinates
                    c = selection.center["xyz".index(axis)]
                    ends = [c + e for e in ends]
                box = tuple(QtWidgets.QLineEdit(show_value(e) if e != "" else "") for e in ends)
                self._along_rows = []
                for b, label in zip(box, FORM[kind][2]["along"]):
                    b.setPlaceholderText("optional")
                    row = self._beside(b, self.length_unit)
                    self.form.addRow(label, row)
                    self._along_rows.append(row)
                self.inputs[field] = box
                for b in box:
                    b.textChanged.connect(self.changed)
                self.inputs["axis"].currentTextChanged.connect(self._label_along)
                self._label_along()
                continue
            how = field_kind(cls, field)
            if how.choices:
                options = how.choices
                box = QtWidgets.QComboBox()
                box.setEditable(how.editable)
                box.setInsertPolicy(QtWidgets.QComboBox.NoInsert)      # Enter previews; it adds no choice
                for shown, v in options:
                    box.addItem(shown, v)
                text = show_value(value if value is not None else options[0][1], options)
                if box.findText(text) < 0:
                    box.addItem(text, None)
                box.setCurrentIndex(box.findText(text))
                widget = box
            elif how.kind == "tick":
                box = QtWidgets.QCheckBox()
                box.setChecked(bool(value))
                widget = box
            elif how.kind == "file":
                box = QtWidgets.QLineEdit("" if value is None else str(value))
                widget = QtWidgets.QWidget()
                row = QtWidgets.QHBoxLayout(widget)
                row.setContentsMargins(0, 0, 0, 0)
                row.addWidget(box)
                browse = QtWidgets.QPushButton("Browse...")
                browse.clicked.connect(lambda _=False, b=box, f=how.files: self._browse(b, f))
                row.addWidget(browse)
            else:
                box = QtWidgets.QLineEdit(show_value(value))
                widget = box
                if how.kind == "point":
                    box.setPlaceholderText("x, y, z")
                if how.optional:
                    box.setPlaceholderText("automatic")
            unit = self.length_unit if how.kind in ("length", "point") else ""
            info = TOLERANCE[1] if field == "pad" else how.help
            if unit or info:
                widget = self._beside(widget, unit, info)
            self.form.addRow(field_label(kind, field), widget)
            self.inputs[field] = box
            for signal in (("textChanged", "currentTextChanged", "toggled")):
                if hasattr(box, signal):
                    getattr(box, signal).connect(self.changed)
                    break
        if kind in CLICK_TO_FILL and self.main is not None:
            text, info, fill = CLICK_TO_FILL[kind]
            button = QtWidgets.QPushButton(text)
            button.setAutoDefault(False)
            button.clicked.connect(lambda _=False, f=fill: self._wait_for_click(f))
            self.form.addRow(self._beside(button, "", info))
        QtCore.QTimer.singleShot(0, self._fit)

    def _wait_for_click(self, fill):
        """The next click on the part fills the form in (CLICK_TO_FILL), from the face it lands on."""
        def take(cell):
            fill(self, *self.main.wall_triangles(cell))
        self.on_click = take
        self.result.setText("Click the inside wall of the hole on the part.")

    def _fit(self):
        """Tall enough for every row of the form: a long form is not squashed."""
        self.resize(self.width(), max(self.height(), self.sizeHint().height()))

    def showEvent(self, event):
        super().showEvent(event)
        self._fit()

    def _cylinder_axis(self):
        """x, y or z when the cylinder's axis is one of them, else None (a tilted axis)."""
        axis = self._axis_value()
        return axis if axis in ("x", "y", "z") else None

    def _axis_value(self):
        """The axis box's value: x, y, z, or what was typed in it (a direction such as 1, 1, 0)."""
        box = self.inputs.get("axis")
        if box is None:
            return ""
        text = box.currentText().strip()
        index = box.findText(text)
        return (box.itemData(index) if index >= 0 and box.itemData(index) else text).strip()

    def _label_along(self, *_):
        """Cylinder: the ends as coordinates (x from / x to) for an x, y or z axis, else distances."""
        axis = self._cylinder_axis()
        texts = (f"{axis} from", f"{axis} to") if axis else ("From (distance)", "To (distance)")
        for row, text in zip(getattr(self, "_along_rows", []), texts):
            label = self.form.labelForField(row)
            if label is not None:
                label.setText(text)

    def _show_limits(self, *_):
        """Flat cut: from / to rows for the two axes the cut is not across."""
        limits = self.inputs.get("limits")
        if not limits:
            return
        cut = self._axis_value()
        for a, (pair, row) in limits.items():
            row.setVisible(a != cut)
            label = self.form.labelForField(row)
            if label is not None:
                label.setVisible(a != cut)

    @staticmethod
    def _beside(widget, unit, info=None):
        """widget with its unit, and an (i), after it."""
        box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(widget, stretch=1)
        if unit:
            label = QtWidgets.QLabel(unit)
            label.setStyleSheet(GREY)
            row.addWidget(label)
        if info:
            row.addWidget(Info(info))
        return box

    def _browse(self, box, pattern):
        from .pyto_ad_imports import FILE_DIALOG_OPTIONS
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose a file", self.project.base_dir, pattern,
                                                        options=FILE_DIALOG_OPTIONS)
        if path:
            try:
                path = os.path.relpath(path, self.project.base_dir).replace("\\", "/")
            except ValueError:                                      # another drive on Windows
                pass
            box.setText(path)

    def _switch(self, index):
        """Form to code line: the code line shows the form's selection, when it is complete."""
        if index == 1:
            try:
                self.code.setText(repr(self.from_form()))
            except ValueError:
                pass

    @staticmethod
    def _value(label, text):
        """A typed value: a number, a point (0, 38, 0, brackets optional), a list, a word such as xmax, or
        a length in element sizes h (1.5h, 5 + 1h), kept as text; a number may be plain arithmetic (38/2,
        2*pi*5, also inside a point: 0, 38/2, 0)."""
        from lsr_mmto.problem import selections as S
        try:
            value = plain_value(ast.parse(text, mode="eval").body)
        except (ValueError, SyntaxError):
            if text.replace("_", "").isalnum():                             # xmax without quotes, 2h
                return text
            try:
                S._length_value(text, 1.0)                                  # 1.5h, 5 + 1h: worked out per mesh
                return text
            except ValueError:
                raise ValueError(f"{label}: {text!r} is not a value; write e.g. 2.5 or 0, 38, 0, or a length "
                                 f"in element sizes h such as 5 + 1h") from None
        return list(value) if isinstance(value, tuple) else value

    def from_form(self):
        kind = self.kind()
        if kind == "combined":
            from lsr_mmto.problem import selections as S
            how, a, b = (self.inputs["how"].currentData(), self.inputs["a"].currentText(),
                         self.inputs["b"].currentText())
            if a == b:
                raise ValueError("Selection A and Selection B must be different selections")
            return {"union": S.union, "intersect": S.intersect, "difference": S.difference}[how](
                S.selection(a), S.selection(b))
        cls = self.types[kind]
        kwargs = {}
        for field, box in self.inputs.items():
            label = field_label(kind, field)
            if field == "limits":
                cut = kwargs.get("axis") or self.inputs["axis"].currentData()
                limits = {}
                for a, (pair, _) in box.items():
                    ends = [b.text().strip() for b in pair]
                    if a == cut or not any(ends):
                        continue
                    if not all(ends):
                        raise ValueError(f"{a} from / to: give both, or leave both empty")
                    limits[a] = [self._value(label, e) for e in ends]
                kwargs[field] = limits or None
                continue
            if field == "along":
                ends = [b.text().strip() for b in box]
                if any(ends) and not all(ends):
                    raise ValueError("the two ends: give both, or leave both empty")
                kwargs[field] = [self._value(label, e) for e in ends] if all(ends) else None
                axis = self._cylinder_axis()
                if kwargs[field] and axis and isinstance(kwargs.get("center"), list) and \
                        all(isinstance(e, (int, float)) for e in kwargs[field]):   # coordinates -> from the point
                    c = float(kwargs["center"]["xyz".index(axis)])
                    kwargs[field] = [e - c for e in kwargs[field]]
                continue
            if isinstance(box, QtWidgets.QCheckBox):
                kwargs[field] = box.isChecked()
                continue
            if isinstance(box, QtWidgets.QComboBox):
                text = box.currentText().strip()
                index = box.findText(text)
                if index >= 0 and box.itemData(index) is not None:          # one of the listed choices
                    kwargs[field] = box.itemData(index)
                    continue
            else:
                text = box.text().strip()
            how = field_kind(cls, field)
            if how.kind == "file":
                kwargs[field] = text
                continue
            if not text:
                if how.optional:
                    kwargs[field] = None
                    continue
                raise ValueError(f"{label}: no value")
            kwargs[field] = self._value(label, text)
            if how.kind == "numbers" and isinstance(kwargs[field], int):        # one number: a list of one
                kwargs[field] = [kwargs[field]]
        try:
            return cls(**kwargs)
        except (TypeError, ValueError, KeyError) as e:
            raise ValueError(f"{shown_name(kind)}: {e}") from None

    def selection(self):
        return self.from_form() if self.tabs.currentIndex() == 0 else parse_selection(self.code.text())

    def problems(self):
        out = []
        name = self.name.text().strip()
        if not name.isidentifier():
            out.append(f"name {name!r}: letters, digits and _, not starting with a digit")
        elif name != self.old_name and name in self.project.selections:
            out.append(f"there is already a selection named {name}")
        try:
            from lsr_mmto.problem import selections as S
            selection = self.selection()
            S.resolved(selection, dict(self.project.selections, **{name: selection}), (name,))
        except ValueError as e:
            out.append(str(e))
        return out

    def keyPressEvent(self, event):
        if not preview_on_enter(self, event):
            super().keyPressEvent(event)

    def show_preview(self):
        problems = self.problems()
        if problems:
            self.result.setText("<span style='color:#b00000'>" + "<br>".join(problems) + "</span>")
            return
        if self.preview is not None:
            self.result.setText(self.preview(self.selection()))
            self.shown = True

    def clear_shown(self):
        """This form's preview off the part, and its node count."""
        if self.shown and self.preview is not None:
            self.preview(None)
        self.shown = False
        self.result.setText("")

    def changed(self, *_):
        """An edit: the preview no longer shows this form, so it goes."""
        self.clear_shown()

    def done(self, result):
        self.clear_shown()                                          # OK or Cancel: nothing left behind
        super().done(result)

    def accept(self):
        problems = self.problems()
        if problems:
            QtWidgets.QMessageBox.warning(self, "Selection", "\n".join(problems))
            return
        super().accept()


# ---------------------------------------------------------------------- a support or load
# What the support or load dialog shows for each kind (its name, what it is, a label per value) is the
# kind's own: BCKind.shown, help and the labels of its values, given by its physics.
def _kind(kind):
    from lsr_mmto.problem.boundary_conditions import BC_KINDS
    return BC_KINDS.get(kind)


def bc_name(kind):
    k = _kind(kind)
    return k.shown if k is not None and k.shown else kind


def bc_values(bc, units=None):
    """A support or load's values in words, for the table: e.g. Held in x, y, z."""
    out = []
    kind = _kind(bc["kind"])
    labels = {v.name: v.label for v in kind.values} if kind is not None else {}
    typ = {v.name: v.kind for v in kind.values} if kind is not None else {}
    for k, v in bc.items():
        if k in ("kind", "on"):
            continue
        if typ.get(k) == "dofs":
            text = ", ".join(v)
        elif typ.get(k) == "area" and v == "bbox":
            text = "from the selection"
        else:
            text = show_value(v)
        out.append(f"{labels.get(k, k)} {text}")
    return "; ".join(out)


def bc_value(bc, units=None):
    """A support or load's value, short, for the table: x, y, z / 0, -4, 0 N / 75 MPa. A vector that
    measures nothing is a direction (along ...), last; an area is left out."""
    kind = _kind(bc["kind"])
    fields = {v.name: (v.kind, v.measures) for v in kind.values} if kind is not None else {}
    parts, along = [], []
    for k, v in bc.items():
        typ, measures = fields.get(k, (None, None))
        if k in ("kind", "on") or typ == "area":
            continue
        if typ == "vector" and measures is None:
            along.append(f"along {show_value(v)}")
            continue
        text = ", ".join(v) if typ == "dofs" else show_value(v)
        unit = field_unit(measures, units)
        parts.append(text + (f" {unit}" if unit else ""))
    return "; ".join(parts + along)


def _clear_label(page, name):
    try:
        getattr(page, name).setText("")
    except RuntimeError:                                            # the page has closed
        pass


def field_unit(measures, units):
    from lsr_mmto.materials.databases import UNIT_SYSTEMS
    if measures is None or units not in UNIT_SYSTEMS:
        return ""
    s = UNIT_SYSTEMS[units]
    return f"{s['length']}^2" if measures == "area" else s.get(measures, "")


class BCDialog(QtWidgets.QDialog):
    """A support or load: its kind, its region, and the kind's values (BCKind.values)."""

    def __init__(self, parent, project, bc=None, preview=None):
        from lsr_mmto.problem.boundary_conditions import BC_KINDS
        super().__init__(parent)
        self.preview = preview
        self.pages = None                                           # the strip's open_page, when docked
        self.new_regions = {}                                       # made here with New selection; added on OK
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.setWindowTitle("Support or load")
        self.resize(480, 220)
        self.project, self.old, self.kinds = project, bc, BC_KINDS
        layout = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QFormLayout()
        self.kind = QtWidgets.QComboBox()
        from lsr_mmto.physics.available_physics import PHYSICS
        fields = PHYSICS[project.physics].FIELDS if project.physics in PHYSICS else {}
        for k, kind in BC_KINDS.items():
            if kind.field in fields:                                # the kinds of the project's physics
                self.kind.addItem(bc_name(k), k)
        self.kind_info = Info("")
        self.on = QtWidgets.QComboBox()
        self.on.addItems(list(project.selections))
        kind_row = QtWidgets.QHBoxLayout()
        kind_row.addWidget(self.kind, stretch=1)
        kind_row.addWidget(self.kind_info)
        top.addRow("Type", kind_row)
        where_row = QtWidgets.QHBoxLayout()
        where_row.addWidget(self.on, stretch=1)
        new_place = QtWidgets.QPushButton("New selection...")
        new_place.setToolTip("Describe a new selection: a face of the part, a box, a cylinder, ...")
        new_place.clicked.connect(self.new_shape)
        where_row.addWidget(new_place)
        top.addRow("Where", where_row)
        layout.addLayout(top)
        self.form = QtWidgets.QFormLayout()
        layout.addLayout(self.form)
        layout.addStretch()
        self.result = QtWidgets.QLabel()                            # the check and the preview's node count
        self.result.setWordWrap(True)
        layout.addWidget(self.result)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        show = buttons.addButton("Preview", QtWidgets.QDialogButtonBox.ActionRole)
        show.setToolTip("Check what you typed and show it on the part (Enter)")
        show.clicked.connect(self.show_preview)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        enter_previews(buttons, show)
        layout.addWidget(buttons)
        self.shown = False                                          # a preview of this form is on the part
        self.inputs = {}
        self.kind.currentIndexChanged.connect(lambda _: (self.changed(), self._fill(self.kind.currentData(), None)))
        self.on.currentIndexChanged.connect(self.changed)
        if bc:
            self.kind.blockSignals(True)
            self.kind.setCurrentIndex(self.kind.findData(bc["kind"]))
            self.kind.blockSignals(False)
            self.on.setCurrentText(bc["on"])
            self._fill(bc["kind"], bc)
        else:
            self._fill(self.kind.currentData(), None)

    def _fill(self, kind, bc):
        while self.form.rowCount():
            self.form.removeRow(0)
        self.kind_info.setToolTip(self.kinds[kind].help)
        self.inputs = {}
        for v in self.kinds[kind].values:
            name, typ, measures = v.name, v.kind, v.measures
            value = (bc or {}).get(name)
            unit = field_unit(measures, self.project.units)
            row = QtWidgets.QHBoxLayout()
            if typ == "dofs":
                boxes = {d: QtWidgets.QCheckBox(d) for d in "xyz"}
                for d, b in boxes.items():
                    b.setChecked(d in (value or "xyz"))
                    row.addWidget(b)
                widget = boxes
            elif typ == "vector":
                widget = [NumberEdit((value or [0.0, 0.0, 0.0])[i], arithmetic=True) for i in range(3)]
                for label, w in zip("xyz", widget):
                    row.addWidget(QtWidgets.QLabel(label))
                    row.addWidget(w)
            elif typ == "area":
                widget = (QtWidgets.QCheckBox("from the selection"),
                          NumberEdit(0.0 if value in (None, "bbox") else value, arithmetic=True))
                widget[0].setChecked(value in (None, "bbox"))
                widget[1].setEnabled(value not in (None, "bbox"))
                widget[0].toggled.connect(lambda on, w=widget[1]: w.setEnabled(not on))
                row.addWidget(widget[0])
                row.addWidget(widget[1])
            else:
                widget = NumberEdit(0.0 if value is None else value, arithmetic=True)
                row.addWidget(widget)
            if unit:
                u = QtWidgets.QLabel(unit)
                u.setStyleSheet(GREY)
                row.addWidget(u)
            label = QtWidgets.QLabel(v.label)
            self.form.addRow(label, row)
            self.inputs[name] = (typ, widget)
            for w in (widget.values() if isinstance(widget, dict) else widget if isinstance(widget, (list, tuple))
                      else [widget]):
                (w.toggled if isinstance(w, QtWidgets.QCheckBox) else w.textChanged).connect(self.changed)

    def new_shape(self):
        """A new selection from the selection form, a page over this one when docked; it joins the project when
        this dialog is closed with OK."""
        q = self.project.copy()
        q.selections.update(self.new_regions)
        dialog = RegionDialog(self, q, preview=self.preview)

        def ok(d):
            name = d.name.text().strip()
            self.new_regions[name] = d.selection()
            self.on.addItem(name)
            self.on.setCurrentText(name)
        if self.pages is not None:
            return self.pages(dialog, ok)
        if dialog.exec_():
            ok(dialog)
        return dialog

    def bc(self):
        """The support or load as the dialog shows it, each unchanged value exactly as it was."""
        kind = self.kind.currentData()
        out = {"kind": kind, "on": self.on.currentText()}
        for name, (typ, widget) in self.inputs.items():
            if typ == "dofs":
                out[name] = "".join(d for d, b in widget.items() if b.isChecked())
            elif typ == "vector":
                out[name] = [w.value() for w in widget]
            elif typ == "area":
                out[name] = "bbox" if widget[0].isChecked() else widget[1].value()
            else:
                out[name] = widget.value()
        old = self.old or {}
        if old.get("kind") == kind:
            for k, v in out.items():
                if k in old and (old[k] == v or (k not in ("kind", "on") and _same(old[k], v))):
                    out[k] = old[k]
        return out

    def problems(self):
        out = []
        if not self.on.currentText():
            out.append("Where: choose a selection, or make one with New selection")
        labels = {v.name: v.label for v in self.kinds[self.kind.currentData()].values}
        for name, (typ, widget) in self.inputs.items():
            name = labels.get(name, name)
            boxes = widget if typ == "vector" else [widget[1]] if typ == "area" and not widget[0].isChecked() \
                else [widget] if typ == "number" else []
            if any(not b.ok() for b in boxes):
                out.append(f"{name} is not a number")
            if typ == "dofs" and not any(b.isChecked() for b in widget.values()):
                out.append(f"{name}: tick at least one of x, y, z")
        if not out:
            out += self.kinds[self.kind.currentData()].errors(self.bc())
        return out

    def keyPressEvent(self, event):
        if not preview_on_enter(self, event):
            super().keyPressEvent(event)

    def show_preview(self):
        """The support or load checked; with no mistake, its selection on the part (and a load's arrow)."""
        problems = self.problems()
        if problems:
            self.result.setText("<span style='color:#b00000'>" + "<br>".join(problems) + "</span>")
            return
        if self.preview is None:
            self.result.setText("")
            return
        name = self.on.currentText()
        selection = self.new_regions.get(name) or self.project.selections[name]
        try:
            self.result.setText(self.preview(selection, self.bc()))
            self.shown = True
        except Exception as e:                                      # noqa: BLE001
            self.result.setText(f"<span style='color:#b00000'>Error: {e}</span>")

    def changed(self, *_):
        """An edit: the preview no longer shows this form, so it goes."""
        if self.shown and self.preview is not None:
            self.preview(None)
        self.shown = False
        self.result.setText("")

    def done(self, result):
        self.changed()                                              # OK or Cancel: nothing left behind
        super().done(result)

    def accept(self):
        problems = self.problems()
        if problems:
            QtWidgets.QMessageBox.warning(self, "Support or load", "\n".join(problems))
            return
        super().accept()


def _same(a, b):
    try:
        return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=0, atol=0)
    except (TypeError, ValueError):
        return a == b


# ---------------------------------------------------------------------- the panel
class SupportsPanel(QtWidgets.QDialog):
    """The supports and loads, added by clicking on the part or by hand; the selections they are on (closed
    until asked for), with keep solid; and the preview. Changes go into the project at once."""

    def __init__(self, window):
        super().__init__(window)
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setWindowTitle("Supports and loads")
        self.resize(760, 620)
        self.setMinimumWidth(360)
        self.window = window
        self.counts = {}                                            # region -> nodes, after a preview
        outer = QtWidgets.QVBoxLayout(self)                         # pages, one at a time: the list first,
        outer.setContentsMargins(0, 0, 0, 0)                        # then any form opened from it on top
        self.stack = QtWidgets.QStackedWidget()
        outer.addWidget(self.stack)
        home = QtWidgets.QWidget()
        self.stack.addWidget(home)
        layout = QtWidgets.QVBoxLayout(home)

        top = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add")
        menu = QtWidgets.QMenu(add)                                 # both ways, at the moment of adding
        menu.addAction("Click on the part...", self.pick)
        menu.addAction("Enter by hand...", self.add_bc)
        add.setMenu(menu)
        top.addWidget(add)
        for text, slot, tip in (("Preview", self.preview, "Show every support and load on the part"),
                                ("Clear", self.clear_preview, "Clear the preview and any unapplied picks")):
            button = QtWidgets.QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            top.addWidget(button)
        top.addStretch()
        layout.addLayout(top)

        self.bcs = QtWidgets.QTableWidget(0, 3)                     # the supports and loads: the main list
        self.bcs.setHorizontalHeaderLabels(["Type", "Where", "Value"])
        self.bcs.horizontalHeaderItem(1).setToolTip("The selection it acts on, by name; Selections, below, lists them all")
        self.bcs.verticalHeader().setVisible(False)
        self.bcs.setToolTip("Double-click a support or load to edit it")
        self.bcs.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.bcs.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.bcs.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        for c in range(3):
            self.bcs.horizontalHeader().setSectionResizeMode(
                c, QtWidgets.QHeaderView.Stretch if c == 2 else QtWidgets.QHeaderView.ResizeToContents)
        self.bcs.doubleClicked.connect(lambda _: self.edit_bc())
        self.bcs.itemSelectionChanged.connect(self.show_selected_bc)
        self.bcs.setMinimumHeight(120)
        self.empty = QtWidgets.QLabel("Nothing yet. Use Add to put a support or load on the part.")
        self.empty.setStyleSheet(GREY)
        self.empty.setWordWrap(True)
        self.empty.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        layout.addWidget(self.empty)
        layout.addWidget(self.bcs, stretch=1)
        delete_row = self._buttons((("Delete", self.delete_bc),))
        self.delete_bc_button = delete_row.itemAt(0).widget()
        layout.addLayout(delete_row)

        self.places_header = QtWidgets.QToolButton()               # the places: closed until asked for
        self.places_header.setCheckable(True)
        self.places_header.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.places_header.setArrowType(QtCore.Qt.RightArrow)
        self.places_header.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        places = QtWidgets.QWidget()
        places_layout = QtWidgets.QVBoxLayout(places)
        places_layout.setContentsMargins(0, 0, 0, 0)
        places.setVisible(False)
        self.places_header.toggled.connect(lambda on: (places.setVisible(on), self.places_header.setArrowType(
            QtCore.Qt.DownArrow if on else QtCore.Qt.RightArrow)))
        places_row = QtWidgets.QHBoxLayout()
        places_row.addWidget(self.places_header)
        places_row.addWidget(Info(PLACES_HELP))
        places_row.addStretch()
        layout.addLayout(places_row)
        self.regions = QtWidgets.QTableWidget(0, 3)
        self.regions.setHorizontalHeaderLabels(["Name", "Type", "Keep solid"])
        self.regions.verticalHeader().setVisible(False)
        self.regions.setToolTip("Double-click a selection to edit or rename it")
        self.regions.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.regions.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.regions.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        for c in range(3):
            self.regions.horizontalHeader().setSectionResizeMode(
                c, QtWidgets.QHeaderView.Stretch if c == 1 else QtWidgets.QHeaderView.ResizeToContents)
        self.regions.doubleClicked.connect(lambda _: self.edit_region())
        self.regions.itemSelectionChanged.connect(self.show_selected_place)
        self.regions.setMinimumHeight(120)
        places_layout.addWidget(self.regions, stretch=1)
        places_layout.addLayout(self._buttons((("Add", self.add_region), ("Delete", self.delete_region))))
        layout.addWidget(places, stretch=1)

        self.report = QtWidgets.QLabel()                            # one line after a preview; details in (i)
        self.report.setWordWrap(True)
        self.report_info = Info("")
        self.report_info.setVisible(False)
        report_row = QtWidgets.QHBoxLayout()
        report_row.addWidget(self.report, stretch=1)
        report_row.addWidget(self.report_info, alignment=QtCore.Qt.AlignTop)
        layout.addLayout(report_row)
        layout.addStretch()                                         # an empty list: everything stays on top
        self.refresh()

    # ------------------------------------------------------------------ pages
    def open_page(self, dialog, on_ok):
        """dialog as a page of this strip, in front of the page before it, the 3D view free to use;
        OK or Cancel goes back one page, OK first calls on_ok(dialog)."""
        page = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(page)
        title = QtWidgets.QLabel(f"<b>{dialog.windowTitle()}</b>")
        page.mouse = QtWidgets.QLabel(MOUSE_HINT)                   # the point under the mouse, while here
        page.mouse.setStyleSheet(GREY)
        page.copied = QtWidgets.QLabel("")                          # a point copied by a click, for a while
        page.copied.setStyleSheet("color: #1a7f37;")
        dialog.setWindowFlags(QtCore.Qt.Widget)
        for widget in (title, dialog, page.mouse, page.copied):
            box.addWidget(widget)
        box.setStretch(1, 1)
        self.stack.addWidget(page)
        self.stack.setCurrentWidget(page)
        dialog.show()

        def finished(result):
            if result:
                on_ok(dialog)
            self.stack.removeWidget(page)
            page.deleteLater()
        dialog.finished.connect(finished)
        return dialog

    def take_click(self, cell):
        """A click on triangle cell of the part, for the form in front when it waits for one (Click a hole):
        True when the form took it."""
        form = self.stack.currentWidget().findChild(QtWidgets.QDialog)
        take = getattr(form, "on_click", None)
        if take is None:
            return False
        form.on_click = None
        take(cell)
        return True

    def show_copied(self, text):
        """A click on the part while a form is open: what went to the clipboard."""
        page = self.stack.currentWidget()
        if hasattr(page, "copied"):
            page.copied.setText(f"Copied {text}: paste it with Ctrl+V")
            QtCore.QTimer.singleShot(5000, lambda p=page: _clear_label(p, "copied"))

    def show_mouse(self, text):
        """The point under the mouse, on the page in front (the list has the status bar for it)."""
        page = self.stack.currentWidget()
        if hasattr(page, "mouse"):
            page.mouse.setText(f"Mouse on the part: {text}" if text else MOUSE_HINT)

    @staticmethod
    def _buttons(spec):
        row = QtWidgets.QHBoxLayout()
        for text, slot in spec:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda _=False, s=slot: s())
            row.addWidget(button)
        row.addStretch()
        return row

    @property
    def project(self):
        return self.window.project

    # ------------------------------------------------------------------ the tables
    def refresh(self):
        p = self.project
        names = list(p.selections)
        self.regions.setRowCount(len(names))
        for r, name in enumerate(names):
            self.regions.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
            selection = p.selections[name]
            item = QtWidgets.QTableWidgetItem(shown_name(selection.TYPE) if selection.TYPE in FORM else "Combined")
            item.setToolTip(describe(selection) + (f"\n{self.counts[name]} nodes" if name in self.counts else ""))
            self.regions.setItem(r, 1, item)
            keep = QtWidgets.QCheckBox()
            keep.setChecked(name in p.keep_solid_names)
            keep.setToolTip("Keep this selection solid: the optimization does not remove it")
            keep.toggled.connect(lambda on, n=name: self.set_keep_solid(n, on))
            cell = QtWidgets.QWidget()
            centred = QtWidgets.QHBoxLayout(cell)
            centred.setContentsMargins(0, 0, 0, 0)
            centred.addWidget(keep, alignment=QtCore.Qt.AlignCenter)
            self.regions.setCellWidget(r, 2, cell)
        self.places_header.setText(f"Selections ({len(names)})")
        self.empty.setVisible(not p.bcs)
        self.bcs.setVisible(bool(p.bcs))
        self.delete_bc_button.setVisible(bool(p.bcs))
        self.bcs.setRowCount(len(p.bcs))
        for r, bc in enumerate(p.bcs):
            for c, text in enumerate((bc_name(bc["kind"]), bc["on"], bc_value(bc, p.units))):
                item = QtWidgets.QTableWidgetItem(text)
                where = p.selections.get(bc["on"])
                item.setToolTip(f"{bc['on']} is a selection: {describe(where)}. Selections, below, lists them all."
                                if c == 1 and where is not None else text)
                self.bcs.setItem(r, c, item)
        self.window.project_changed()

    def _selected(self, table):
        rows = table.selectionModel().selectedRows()
        return rows[0].row() if rows else None

    def _region(self):
        r = self._selected(self.regions)
        return None if r is None else list(self.project.selections)[r]

    # ------------------------------------------------------------------ regions
    def add_region(self):
        def ok(dialog):
            self.project.add_selection(dialog.name.text().strip(), dialog.selection())
            self.after_change()
        return self.open_page(RegionDialog(self, self.project, preview=self.preview_one), ok)

    def edit_region(self, name=None):
        name = name or self._region()
        if name is None:
            return None
        def ok(dialog):
            new = dialog.name.text().strip()
            selection = dialog.selection()
            if new != name:
                self.rename(name, new)
            if repr(selection) != repr(self.project.selections[new]):
                self.project.selections[new] = selection
            self.after_change()
        return self.open_page(RegionDialog(self, self.project, name, preview=self.preview_one), ok)

    def rename_region(self):
        name = self._region()
        if name is None:
            return
        new, ok = QtWidgets.QInputDialog.getText(self, "Rename selection", "New name", text=name)
        new = new.strip()
        if ok and new and new != name:
            if not new.isidentifier() or new in self.project.selections:
                QtWidgets.QMessageBox.warning(self, "Rename selection", f"{new!r}: not a free name made of "
                                                                     f"letters, digits and _")
                return
            self.rename(name, new)
            self.after_change()

    def rename(self, old, new):
        """The selection and everything that names it, combined selections too."""
        self.project.rename_selection(old, new)
        if old in self.counts:
            self.counts[new] = self.counts.pop(old)

    def delete_region(self, name=None):
        name = name or self._region()
        if name is None:
            return False
        users = used_by(self.project, name)
        if users:
            QtWidgets.QMessageBox.warning(self, "Delete selection", f"{name} is used by {', '.join(users)}; "
                                                                 f"remove that first.")
            return False
        del self.project.selections[name]
        self.counts.pop(name, None)
        self.after_change()
        return True

    def set_keep_solid(self, name, on):
        p = self.project
        if on and name not in p.keep_solid_names:
            p.keep_solid_names.append(name)
        elif not on and name in p.keep_solid_names:
            p.keep_solid_names.remove(name)
        self.after_change()

    # ------------------------------------------------------------------ supports and loads
    def add_bc(self):
        def ok(dialog):
            for name, selection in dialog.new_regions.items():
                self.project.add_selection(name, selection)
            self.project.bcs.append(dialog.bc())
            self.after_change()
            self.bcs.selectRow(len(self.project.bcs) - 1)
        dialog = BCDialog(self, self.project, preview=self.preview_one)
        dialog.pages = self.open_page
        return self.open_page(dialog, ok)

    def edit_bc(self, index=None):
        index = self._selected(self.bcs) if index is None else index
        if index is None:
            return None
        def ok(dialog):
            for name, selection in dialog.new_regions.items():
                self.project.add_selection(name, selection)
            self.project.bcs[index] = dialog.bc()
            self.after_change()
        dialog = BCDialog(self, self.project, self.project.bcs[index], preview=self.preview_one)
        dialog.pages = self.open_page
        return self.open_page(dialog, ok)

    def delete_bc(self, index=None):
        index = self._selected(self.bcs) if index is None else index
        if index is not None:
            del self.project.bcs[index]
            self.after_change()

    # ------------------------------------------------------------------ picking and preview
    def pick(self):
        """The Pick window, in place of this list until its Apply adds the pick to the project."""
        self.window.open_structural_loads_window()

    def show_selected_bc(self):
        """Only the support or load clicked, on the part."""
        r = self._selected(self.bcs)
        if r is not None and r < len(self.project.bcs):
            from lsr_mmto.problem.boundary_conditions import BC_KINDS
            bc = self.project.bcs[r]
            colour = SUPPORT_COLOUR if BC_KINDS[bc["kind"]].role == "support" else LOAD_COLOUR
            self._show_one(bc["on"], colour, r)

    def show_selected_place(self):
        """Only the selection clicked, on the part."""
        name = self._region()
        if name is not None:
            self._show_one(name, OTHER_COLOUR, None)

    def _show_one(self, name, colour, bc_index):
        try:
            n = self.window.draw_one(name, colour, bc_index)
            self.report.setText(f"{name}: {n} nodes" if n else
                                f"<span style='color:#b00000'>{name} has no nodes on this mesh</span>")
        except Exception as e:                                      # noqa: BLE001
            self.report.setText(f"<span style='color:#b00000'>Error: {e}</span>")
        self.report_info.setVisible(False)

    def after_change(self):
        """The lists again; a preview showing is redone, any other result line goes (it is out of date)."""
        self.refresh()
        if self.window.preview_on:
            self.preview()
        else:                                                       # a single row shown, or nothing: cleared
            self.window.clear_preview()
            self.report.setText("")
            self.report_info.setVisible(False)

    def preview_one(self, selection, bc=None):
        """A region being edited, on the run's mesh: its node count, drawn in orange; with bc, the support
        or load on it being edited: blue for a support, red with its arrow for a load. None clears it."""
        from lsr_mmto.problem.boundary_conditions import BC_KINDS
        from .main_window import load_direction
        self.window.show_setup_view()
        for name in ("mmto_region", "mmto_region_outline", "mmto_arrow_region"):
            self.window.plotter.remove_actor(name, reset_camera=False)
        if selection is None:
            self.window.plotter.render()
            return ""
        mesh = self.window.preview_mesh()
        nodes = self.project.evaluate(selection, mesh, "nodes")
        xyz = np.asarray(mesh.node_xyz, dtype=float)
        colour = OTHER_COLOUR if bc is None else \
            SUPPORT_COLOUR if BC_KINDS[bc["kind"]].role == "support" else LOAD_COLOUR
        self.window.draw_points("mmto_region", xyz[nodes], colour)
        self.window.draw_outline("mmto_region_outline", selection, colour)
        d = load_direction(bc) if bc is not None else None
        if d is not None and len(nodes) and np.linalg.norm(d) > 0:
            self.window._arrow("_region", d, xyz[nodes], float(np.linalg.norm(np.ptp(xyz, axis=0))))
        self.window.plotter.render()
        return f"{len(nodes)} nodes on the run's mesh ({mesh.num_elems} elements)."

    def preview(self):
        """Every region, support and load on the run's mesh, and the rigid-body check: its verdict on one
        line, the rest (mesh size, node counts, keep-solid elements) behind the (i)."""
        self.report.setText("")
        try:
            lines, self.counts = self.window.draw_supports_and_loads()
        except Exception as e:                                       # noqa: BLE001
            lines = [f"<span style='color:#b00000'>Error: {e}</span>"]
        problems = [line for line in lines[:-1] if "#b00000" in line]  # a region with no nodes: on view too
        self.report.setText("<br>".join(problems + [lines[-1]]))
        details = [line for line in lines[:-1] if line not in problems] + \
                  [f"{name}: {n} nodes" for name, n in self.counts.items()]
        self.report_info.setToolTip("<br>".join(details))
        self.report_info.setVisible(bool(details))
        self.refresh()

    def clear_preview(self):
        """The preview, and any faces or nodes picked but not applied."""
        self.window.clear_preview()
        self.window.clear_picks()
        self.counts = {}
        self.report.setText("")
        self.report_info.setVisible(False)
        self.refresh()
