"""
supports.py
===========
The Supports and loads panel: the project's regions (selections), its supports and loads,
and which regions are kept solid, with a preview on the mesh the run will use.

A region is made three ways: picked with the mouse in PyTO-AD's Structural Loads window (each
Apply is adopted into the project at once, by Project.adopt_picks), filled in on a form made
from the selection type's own fields, or typed as a code line, e.g.
rows(face="xmax", axis="y", n=8, end="max"). A code line is read, never run: only selection
types with plain numbers, text and lists are accepted.

The panel works on the project directly (it stays open while you pick); File > Save keeps the
changes, and closing without saving drops them.
"""
import ast
import inspect
import os

import numpy as np
from PyQt5 import QtCore, QtWidgets

from .panels import GREY, Info, NumberEdit

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
              "Cut across y, At 38.", {"axis": "Cut across", "value": "At"}),
    "box": ("Box", "Nodes inside a box given by two opposite corners.", {"lo": "Corner 1", "hi": "Corner 2"}),
    "sphere": ("Sphere", "Nodes within a radius of a point.", {"center": "Centre", "radius": "Radius"}),
    "cylinder": ("Cylinder", "Nodes inside a cylinder, or a tube if you give an inner radius. Good for holes "
                 "and bolt bores. Leave Length from and Length to empty for a cylinder through the whole part; "
                 "otherwise they set its ends, measured along the axis from the point on the axis.",
                 {"center": "Point on the axis", "axis": "Axis direction", "radius": "Radius",
                  "inner_radius": "Inner radius (0 = solid)", "along": ("Length from", "Length to")}),
    "nearest": ("Closest node", "The one node closest to a point.",
                {"point": "Point", "boundary": "Surface nodes only"}),
    "inside_stl": ("Inside a shape", "Nodes inside a closed shape loaded from an STL file.", {"path": "File"}),
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
TOLERANCE = ("Tolerance", "How far outside the shape a node may sit and still count. Automatic is right "
                          "almost always; or type a length.")
PLACES_HELP = ("A place is the part of the model a support or load acts on: faces you clicked, or a shape you "
               "described (a face of the part, a box, a cylinder...). Each place has a name, like clamp or "
               "corner, which shows under Where, so several supports and loads can share one place. Open Places "
               "to rename a place, change it, or tick Keep solid so the optimization never removes it.")
LENGTHS = {"radius", "inner_radius", "distance", "value", "pad", "along"}   # a length unit beside them
POINTS = {"center", "lo", "hi", "point"}                    # x, y, z
OPTIONAL = {"along", "tol"}                                 # empty means None

# Fields with a fixed set of values: a dropdown of (shown, value); editable: a value of your own may be typed
SIDES = [("Left (x min)", "xmin"), ("Right (x max)", "xmax"), ("Bottom (y min)", "ymin"),
         ("Top (y max)", "ymax"), ("Back (z min)", "zmin"), ("Front (z max)", "zmax")]
XYZ = [("x", "x"), ("y", "y"), ("z", "z")]
CHOICES = {"side": (SIDES, False), "face": (SIDES, False), "end": ([("High end", "max"), ("Low end", "min")], False),
           "axis": (XYZ, True),                             # cylinder: also a direction, e.g. 1, 1, 0
           "pad": ([("Automatic", "auto")], True)}          # or a length
TICKS = {"boundary"}                                        # True or False
FILES = {"path": "STL files (*.stl *.STL)"}


# ---------------------------------------------------------------------- regions as text
def selection_types():
    """{type name: class} of the selection types the GUI offers: all but function, which runs a
    Python file when it is evaluated (scripts can still use it)."""
    import selections as S  # type: ignore
    return {k: v for k, v in sorted(S._TYPES.items()) if k != "function"}


def parse_selection(text):
    """A Selection from a code line such as box(lo=[0, 0, 0], hi=[1, 2, 1]). Only selection types,
    with numbers, text, lists and other selections as arguments; anything else is refused."""
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
            return ast.literal_eval(node)
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


def show_value(field, value):
    """A field's value as the form and the tables show it: sides by name, points without brackets."""
    for shown, v in CHOICES.get(field, ([], False))[0]:
        if value == v:
            return shown
    if value is None:
        return ""
    if isinstance(value, (list, tuple)) and all(isinstance(x, (int, float)) for x in value):
        return ", ".join(format(x, ".12g") for x in value)              # 0, -4, 0; 12 digits: kept exactly
    return value if isinstance(value, str) else repr(value)


def describe(selection):
    """A region in words, for the regions table: e.g. Face of the part: Side Right (x max)."""
    kind = selection.TYPE
    if kind not in FORM:
        return repr(selection)
    defaults = dict(type_fields(type(selection)))
    parts = []
    for field in selection.fields:
        value = getattr(selection, field)
        if value == defaults.get(field) or value is None:
            continue
        if field == "along":
            parts.append(f"Length from {show_value('', value[0])} to {show_value('', value[1])}")
        else:
            parts.append(f"{field_label(kind, field)} {show_value(field, value)}")
    return f"{shown_name(kind)}: " + ", ".join(parts)


def type_fields(cls):
    """[(field, default or inspect.Parameter.empty)] of a selection type, from its __init__."""
    params = inspect.signature(cls.__init__).parameters
    return [(f, params[f].default if f in params else inspect.Parameter.empty) for f in cls.fields]


def used_by(project, name):
    """What uses the region: its supports and loads, and keep solid."""
    out = [f"{bc_name(bc['kind'])} {i}" for i, bc in enumerate(project.bcs, 1) if bc.get("on") == name]
    return out + (["keep solid"] if name in project.keep_solid_names else [])


# ---------------------------------------------------------------------- a region
class RegionDialog(QtWidgets.QDialog):
    """A region: its name, and its definition on a form or as a code line."""

    def __init__(self, parent, project, name=None, preview=None):
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.setWindowTitle("Region")
        self.resize(520, 380)
        self.project, self.old_name, self.preview = project, name, preview
        layout = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Name"))
        self.name = QtWidgets.QLineEdit(name or project._free_name("region"))
        row.addWidget(self.name)
        layout.addLayout(row)

        self.tabs = QtWidgets.QTabWidget()
        form_page = QtWidgets.QWidget()
        form_layout = QtWidgets.QVBoxLayout(form_page)
        self.type = QtWidgets.QComboBox()
        current = project.selections.get(name)
        kept = {current.TYPE} if current is not None else set()      # a picked region: its own type too
        self.types = {k: c for k, c in selection_types().items() if k in FORM and (k not in PICKED or k in kept)}
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
        hint = QtWidgets.QLabel("For example rows(face=\"xmax\", axis=\"y\", n=8, end=\"max\"), or regions "
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
        show.clicked.connect(self.show_preview)
        clear = buttons.addButton("Clear", QtWidgets.QDialogButtonBox.ActionRole)
        clear.setToolTip("Clear this preview from the part")
        clear.clicked.connect(self.clear_shown)
        self.shown = False                                          # a preview of this form is on the part
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.inputs = {}
        self.type.currentIndexChanged.connect(lambda _: (self.changed(), self._fill_form(self.kind())))
        if current is not None and current.TYPE in self.types:
            self.type.blockSignals(True)
            self.type.setCurrentIndex(self.type.findData(current.TYPE))
            self.type.blockSignals(False)
            self._fill_form(current.TYPE, current)
        else:
            self._fill_form(self.kind())
        if current is not None:
            self.code.setText(repr(current))
            if current.TYPE not in self.types:
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
        for field, default in type_fields(cls):
            value = getattr(selection, field) if selection is not None else \
                None if default is inspect.Parameter.empty else default
            if field == "along":                                    # two boxes: the ends
                ends = value or ("", "")
                box = tuple(QtWidgets.QLineEdit(show_value("", e) if e != "" else "") for e in ends)
                for b, label in zip(box, FORM[kind][2]["along"]):
                    b.setPlaceholderText("optional")
                    self.form.addRow(label, self._beside(b, self.length_unit))
                self.inputs[field] = box
                for b in box:
                    b.textChanged.connect(self.changed)
                continue
            if field in CHOICES:
                options, editable = CHOICES[field]
                box = QtWidgets.QComboBox()
                box.setEditable(editable)
                for shown, v in options:
                    box.addItem(shown, v)
                text = show_value(field, value if value is not None else options[0][1])
                if box.findText(text) < 0:
                    box.addItem(text, None)
                box.setCurrentIndex(box.findText(text))
                widget = box
            elif field in TICKS:
                box = QtWidgets.QCheckBox()
                box.setChecked(bool(value))
                widget = box
            elif field in FILES:
                box = QtWidgets.QLineEdit("" if value is None else str(value))
                widget = QtWidgets.QWidget()
                row = QtWidgets.QHBoxLayout(widget)
                row.setContentsMargins(0, 0, 0, 0)
                row.addWidget(box)
                browse = QtWidgets.QPushButton("Browse...")
                browse.clicked.connect(lambda _=False, b=box, f=FILES[field]: self._browse(b, f))
                row.addWidget(browse)
            else:
                box = QtWidgets.QLineEdit(show_value(field, value))
                widget = box
                if field in POINTS:
                    box.setPlaceholderText("x, y, z")
                if field in OPTIONAL:
                    box.setPlaceholderText("automatic")
            unit = self.length_unit if field in LENGTHS | POINTS else ""
            if unit or field == "pad":
                widget = self._beside(widget, unit, TOLERANCE[1] if field == "pad" else None)
            self.form.addRow(field_label(kind, field), widget)
            self.inputs[field] = box
            for signal in (("textChanged", "currentTextChanged", "toggled")):
                if hasattr(box, signal):
                    getattr(box, signal).connect(self.changed)
                    break
        QtCore.QTimer.singleShot(0, self._fit)

    def _fit(self):
        """Tall enough for every row of the form: a long form is not squashed."""
        self.resize(self.width(), max(self.height(), self.sizeHint().height()))

    def showEvent(self, event):
        super().showEvent(event)
        self._fit()

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
        from .pyto_ad import FILE_DIALOG_OPTIONS
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
        """A typed value: a number, a point (0, 38, 0, brackets optional), a list, or a word such as 1.5h."""
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            if text.endswith("h") or text.replace("_", "").isalnum():         # "1.5h", xmax without quotes
                return text
            raise ValueError(f"{label}: {text!r} is not a value; write e.g. 2.5 or 0, 38, 0") from None
        return list(value) if isinstance(value, tuple) else value

    def from_form(self):
        kind = self.kind()
        cls = self.types[kind]
        kwargs = {}
        for field, box in self.inputs.items():
            label = field_label(kind, field)
            if field == "along":
                ends = [b.text().strip() for b in box]
                if any(ends) and not all(ends):
                    raise ValueError("Length from and Length to: give both, or leave both empty")
                kwargs[field] = [self._value(label, e) for e in ends] if all(ends) else None
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
            if field in FILES:
                kwargs[field] = text
                continue
            if not text:
                if field in OPTIONAL:
                    kwargs[field] = None
                    continue
                raise ValueError(f"{label}: no value")
            kwargs[field] = self._value(label, text)
            if field in ("triangles", "indices") and isinstance(kwargs[field], int):
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
            out.append(f"there is already a region named {name}")
        try:
            self.selection()
        except ValueError as e:
            out.append(str(e))
        return out

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
            QtWidgets.QMessageBox.warning(self, "Region", "\n".join(problems))
            return
        super().accept()


# ---------------------------------------------------------------------- a support or load
# What the support or load dialog shows for each kind: its name, what it is, and a label per value.
BC_FORM = {
    "fixed": ("Fixed support", "Holds the region still in the ticked directions.", {"dofs": "Held in"}),
    "displacement": ("Moved support", "Moves the region by a set distance in the ticked directions.",
                     {"dofs": "Moved in", "value": "Distance"}),
    "force": ("Force", "A total force on the region, shared equally by its nodes.", {"total": "Total force"}),
    "traction": ("Pressure", "A pressure on the region, turned into a force: Pressure times Area, pushing along "
                 "Direction, shared equally by its nodes. Tick Area from the region to use the region's size "
                 "across the direction (Direction must then be along x, y or z).",
                 {"pressure": "Pressure", "direction": "Direction", "area": "Area"}),
}


def bc_name(kind):
    return BC_FORM[kind][0] if kind in BC_FORM else kind


def bc_values(bc, units=None):
    """A support or load's values in words, for the table: e.g. Held in x, y, z."""
    out = []
    labels = BC_FORM.get(bc["kind"], ("", "", {}))[2]
    for k, v in bc.items():
        if k in ("kind", "on"):
            continue
        if k == "dofs":
            text = ", ".join(v)
        elif v == "bbox":
            text = "from the region"
        else:
            text = show_value("", v)
        out.append(f"{labels.get(k, k)} {text}")
    return "; ".join(out)


def bc_value(bc, units=None):
    """A support or load's value, short, for the table: x, y, z / 0, -4, 0 N / 75 MPa."""
    from project import BC_KINDS  # type: ignore
    fields = {name: measures for name, _, measures, _ in BC_KINDS[bc["kind"]].fields} if bc["kind"] in BC_KINDS \
        else {}
    parts = []
    for k, v in bc.items():
        if k in ("kind", "on") or k == "direction" or k == "area":
            continue
        text = ", ".join(v) if k == "dofs" else show_value("", v)
        unit = field_unit(fields.get(k), units)
        parts.append(text + (f" {unit}" if unit else ""))
    if "direction" in bc:
        parts.append(f"along {show_value('', bc['direction'])}")
    return "; ".join(parts)


def field_unit(measures, units):
    from databases import UNIT_SYSTEMS  # type: ignore
    if measures is None or units not in UNIT_SYSTEMS:
        return ""
    s = UNIT_SYSTEMS[units]
    return f"{s['length']}^2" if measures == "area" else s.get(measures, "")


class BCDialog(QtWidgets.QDialog):
    """A support or load: its kind, its region, and the kind's values (BCKind.fields)."""

    def __init__(self, parent, project, bc=None, preview=None):
        from project import BC_KINDS  # type: ignore
        super().__init__(parent)
        self.preview = preview
        self.new_regions = {}                                       # made here with New shape; added on OK
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.setWindowTitle("Support or load")
        self.resize(480, 220)
        self.project, self.old, self.kinds = project, bc, BC_KINDS
        layout = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QFormLayout()
        self.kind = QtWidgets.QComboBox()
        for k in BC_KINDS:
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
        new_shape = QtWidgets.QPushButton("New shape...")
        new_shape.setToolTip("Describe a new place: a face of the part, a box, a cylinder, ...")
        new_shape.clicked.connect(self.new_shape)
        where_row.addWidget(new_shape)
        top.addRow("Where", where_row)
        layout.addLayout(top)
        self.form = QtWidgets.QFormLayout()
        layout.addLayout(self.form)
        layout.addStretch()
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.inputs = {}
        self.kind.currentIndexChanged.connect(lambda _: self._fill(self.kind.currentData(), None))
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
        self.kind_info.setToolTip(BC_FORM[kind][1] if kind in BC_FORM else self.kinds[kind].doc)
        self.inputs = {}
        for name, typ, measures, doc in self.kinds[kind].fields:
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
                widget = [NumberEdit((value or [0.0, 0.0, 0.0])[i]) for i in range(3)]
                for label, w in zip("xyz", widget):
                    row.addWidget(QtWidgets.QLabel(label))
                    row.addWidget(w)
            elif typ == "area":
                widget = (QtWidgets.QCheckBox("from the region"),
                          NumberEdit(0.0 if value in (None, "bbox") else value))
                widget[0].setChecked(value in (None, "bbox"))
                widget[1].setEnabled(value not in (None, "bbox"))
                widget[0].toggled.connect(lambda on, w=widget[1]: w.setEnabled(not on))
                row.addWidget(widget[0])
                row.addWidget(widget[1])
            else:
                widget = NumberEdit(0.0 if value is None else value)
                row.addWidget(widget)
            if unit:
                u = QtWidgets.QLabel(unit)
                u.setStyleSheet(GREY)
                row.addWidget(u)
            label = QtWidgets.QLabel(BC_FORM.get(kind, ("", "", {}))[2].get(name, name))
            self.form.addRow(label, row)
            self.inputs[name] = (typ, widget)

    def new_shape(self):
        """A new place from the region form; it joins the project when the dialog is closed with OK."""
        q = self.project.copy()
        q.selections.update(self.new_regions)
        dialog = RegionDialog(self, q, preview=self.preview)
        if dialog.exec_():
            name = dialog.name.text().strip()
            self.new_regions[name] = dialog.selection()
            self.on.addItem(name)
            self.on.setCurrentText(name)
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
            out.append("Where: choose a place, or make one with New shape")
        labels = BC_FORM.get(self.kind.currentData(), ("", "", {}))[2]
        for name, (typ, widget) in self.inputs.items():
            name = labels.get(name, name)
            boxes = widget if typ == "vector" else [widget[1]] if typ == "area" and not widget[0].isChecked() \
                else [widget] if typ == "number" else []
            if any(not b.ok() for b in boxes):
                out.append(f"{name} is not a number")
            if typ == "dofs" and not any(b.isChecked() for b in widget.values()):
                out.append(f"{name}: tick at least one of x, y, z")
        if not out:
            out += self.kinds[self.kind.currentData()].check(self.bc())
        return out

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
    """The supports and loads, added by clicking on the part or by hand; the places they are on (closed
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
        layout = QtWidgets.QVBoxLayout(self)

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
        self.bcs.horizontalHeaderItem(1).setToolTip("The place it acts on, by name; Places, below, lists them all")
        self.bcs.verticalHeader().setVisible(False)
        self.bcs.setToolTip("Double-click a support or load to edit it")
        self.bcs.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.bcs.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.bcs.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        for c in range(3):
            self.bcs.horizontalHeader().setSectionResizeMode(
                c, QtWidgets.QHeaderView.Stretch if c == 2 else QtWidgets.QHeaderView.ResizeToContents)
        self.bcs.doubleClicked.connect(lambda _: self.edit_bc())
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
        self.regions.setToolTip("Double-click a place to edit or rename it")
        self.regions.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.regions.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.regions.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        for c in range(3):
            self.regions.horizontalHeader().setSectionResizeMode(
                c, QtWidgets.QHeaderView.Stretch if c == 1 else QtWidgets.QHeaderView.ResizeToContents)
        self.regions.doubleClicked.connect(lambda _: self.edit_region())
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
            keep.setToolTip("Keep this region solid: the optimization does not remove it")
            keep.toggled.connect(lambda on, n=name: self.set_keep_solid(n, on))
            cell = QtWidgets.QWidget()
            centred = QtWidgets.QHBoxLayout(cell)
            centred.setContentsMargins(0, 0, 0, 0)
            centred.addWidget(keep, alignment=QtCore.Qt.AlignCenter)
            self.regions.setCellWidget(r, 2, cell)
        self.places_header.setText(f"Places ({len(names)})")
        self.empty.setVisible(not p.bcs)
        self.bcs.setVisible(bool(p.bcs))
        self.delete_bc_button.setVisible(bool(p.bcs))
        self.bcs.setRowCount(len(p.bcs))
        for r, bc in enumerate(p.bcs):
            for c, text in enumerate((bc_name(bc["kind"]), bc["on"], bc_value(bc, p.units))):
                item = QtWidgets.QTableWidgetItem(text)
                where = p.selections.get(bc["on"])
                item.setToolTip(f"{bc['on']} is a place: {describe(where)}. Places, below, lists them all."
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
        dialog = RegionDialog(self, self.project, preview=self.preview_one)
        if dialog.exec_():
            self.project.add_selection(dialog.name.text().strip(), dialog.selection())
            self.after_change()
        return dialog

    def edit_region(self, name=None):
        name = name or self._region()
        if name is None:
            return None
        dialog = RegionDialog(self, self.project, name, preview=self.preview_one)
        if dialog.exec_():
            new = dialog.name.text().strip()
            selection = dialog.selection()
            if new != name:
                self.rename(name, new)
            if repr(selection) != repr(self.project.selections[new]):
                self.project.selections[new] = selection
            self.after_change()
        return dialog

    def rename_region(self):
        name = self._region()
        if name is None:
            return
        new, ok = QtWidgets.QInputDialog.getText(self, "Rename region", "New name", text=name)
        new = new.strip()
        if ok and new and new != name:
            if not new.isidentifier() or new in self.project.selections:
                QtWidgets.QMessageBox.warning(self, "Rename region", f"{new!r}: not a free name made of "
                                                                     f"letters, digits and _")
                return
            self.rename(name, new)
            self.after_change()

    def rename(self, old, new):
        """The region and everything that names it."""
        p = self.project
        p.selections = {new if k == old else k: v for k, v in p.selections.items()}
        for bc in p.bcs:
            if bc.get("on") == old:
                bc["on"] = new
        p.keep_solid_names = [new if n == old else n for n in p.keep_solid_names]
        if old in self.counts:
            self.counts[new] = self.counts.pop(old)

    def delete_region(self, name=None):
        name = name or self._region()
        if name is None:
            return False
        users = used_by(self.project, name)
        if users:
            QtWidgets.QMessageBox.warning(self, "Delete region", f"{name} is used by {', '.join(users)}; "
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
        dialog = BCDialog(self, self.project, preview=self.preview_one)
        if dialog.exec_():
            for name, selection in dialog.new_regions.items():
                self.project.add_selection(name, selection)
            self.project.bcs.append(dialog.bc())
            self.after_change()
            self.bcs.selectRow(len(self.project.bcs) - 1)
        return dialog

    def edit_bc(self, index=None):
        index = self._selected(self.bcs) if index is None else index
        if index is None:
            return None
        dialog = BCDialog(self, self.project, self.project.bcs[index], preview=self.preview_one)
        if dialog.exec_():
            for name, selection in dialog.new_regions.items():
                self.project.add_selection(name, selection)
            self.project.bcs[index] = dialog.bc()
            self.after_change()
        return dialog

    def delete_bc(self, index=None):
        index = self._selected(self.bcs) if index is None else index
        if index is not None:
            del self.project.bcs[index]
            self.after_change()

    # ------------------------------------------------------------------ picking and preview
    def pick(self):
        """The Pick window, in place of this list until its Apply adds the pick to the project."""
        self.window.open_structural_loads_window()

    def after_change(self):
        self.refresh()
        if self.window.preview_on:
            self.preview()

    def preview_one(self, selection):
        """A region being edited, on the run's mesh: its node count, drawn in orange. None clears it."""
        if selection is None:
            self.window.plotter.remove_actor("mmto_region", reset_camera=False)
            self.window.plotter.render()
            return ""
        mesh = self.window.preview_mesh()
        nodes = selection.evaluate(mesh, "nodes", self.project.base_dir)
        self.window.draw_points("mmto_region", np.asarray(mesh.node_xyz)[nodes], OTHER_COLOUR)
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
