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

from .panels import GREY, NumberEdit

SUPPORT_COLOUR, LOAD_COLOUR, KEEP_COLOUR, OTHER_COLOUR = "#1f5fd6", "#d62020", "#808080", "#e08a00"

# Region form fields with a fixed set of values: a dropdown (editable: a value of your own may be typed too)
SIDES = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
CHOICES = {"side": (SIDES, False), "face": (SIDES, False), "end": (["max", "min"], False),
           "axis": (["x", "y", "z"], True),                 # cylinder: also a direction, e.g. [1, 1, 0]
           "pad": (["auto", "half"], True)}                 # or a length
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


def type_fields(cls):
    """[(field, default or inspect.Parameter.empty)] of a selection type, from its __init__."""
    params = inspect.signature(cls.__init__).parameters
    return [(f, params[f].default if f in params else inspect.Parameter.empty) for f in cls.fields]


def used_by(project, name):
    """What uses the region: its supports and loads, and keep solid."""
    out = [f"{bc['kind']} {i}" for i, bc in enumerate(project.bcs, 1) if bc.get("on") == name]
    return out + (["keep solid"] if name in project.keep_solid_names else [])


# ---------------------------------------------------------------------- a region
class RegionDialog(QtWidgets.QDialog):
    """A region: its name, and its definition on a form or as a code line."""

    def __init__(self, parent, project, name=None, preview=None):
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.setWindowTitle("Region")
        self.resize(640, 360)
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
        self.types = {k: c for k, c in selection_types().items() if c.fields}    # union etc.: the code line
        self.type.addItems(list(self.types))
        form_layout.addWidget(self.type)
        self.doc = QtWidgets.QLabel()
        self.doc.setStyleSheet(GREY)
        self.doc.setWordWrap(True)
        form_layout.addWidget(self.doc)
        self.form = QtWidgets.QFormLayout()
        form_layout.addLayout(self.form)
        note = QtWidgets.QLabel("Values as in Python: 2.5, [0, 38, 0], \"xmax\", \"1.5h\" (1.5 element sizes). "
                                "Union, intersect and difference: use the code line.")
        note.setStyleSheet(GREY)
        note.setWordWrap(True)
        form_layout.addWidget(note)
        form_layout.addStretch()
        self.tabs.addTab(form_page, "Form")
        code_page = QtWidgets.QWidget()
        code_layout = QtWidgets.QVBoxLayout(code_page)
        self.code = QtWidgets.QLineEdit()
        code_layout.addWidget(self.code)
        hint = QtWidgets.QLabel("For example rows(face=\"xmax\", axis=\"y\", n=8, end=\"max\") or "
                                "union(face(\"xmin\"), box(lo=[0, 0, 0], hi=[2, 10, 1]))")
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
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.inputs = {}
        self.type.currentTextChanged.connect(self._fill_form)
        self.tabs.currentChanged.connect(self._switch)
        current = project.selections.get(name)
        if current is not None and current.TYPE in self.types:
            self.type.setCurrentText(current.TYPE)
            self._fill_form(current.TYPE, current)
        else:
            self._fill_form(self.type.currentText())
        if current is not None:
            self.code.setText(repr(current))
            if current.TYPE not in self.types:
                self.tabs.setCurrentIndex(1)

    def _fill_form(self, kind, selection=None):
        while self.form.rowCount():
            self.form.removeRow(0)
        cls = self.types[kind]
        self.doc.setText(" ".join((cls.__doc__ or "").split()))         # the whole description
        self.inputs = {}
        for field, default in type_fields(cls):
            value = getattr(selection, field) if selection is not None else \
                None if default is inspect.Parameter.empty else default
            if field in CHOICES:
                options, editable = CHOICES[field]
                box = QtWidgets.QComboBox()
                box.setEditable(editable)
                box.addItems(options)
                text = value if isinstance(value, str) else options[0] if value is None else repr(value)
                if box.findText(text) < 0:
                    box.addItem(text)
                box.setCurrentText(text)
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
                box = QtWidgets.QLineEdit("" if value is None and default is inspect.Parameter.empty
                                          else repr(value))
                widget = box
            self.form.addRow(field, widget)
            self.inputs[field] = box

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

    def from_form(self):
        cls = self.types[self.type.currentText()]
        kwargs = {}
        for field, box in self.inputs.items():
            if isinstance(box, QtWidgets.QCheckBox):
                kwargs[field] = box.isChecked()
                continue
            text = (box.currentText() if isinstance(box, QtWidgets.QComboBox) else box.text()).strip()
            if field in FILES:
                kwargs[field] = text
                continue
            if not text:
                raise ValueError(f"{field}: no value")
            try:
                kwargs[field] = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                if text.endswith("h") or text.replace("_", "").isalnum():     # "1.5h", xmax typed without quotes
                    kwargs[field] = text
                else:
                    raise ValueError(f"{field}: {text!r} is not a value; write e.g. 2.5, [0, 1, 0] or \"xmax\"") \
                        from None
        try:
            return cls(**kwargs)
        except (TypeError, ValueError, KeyError) as e:
            raise ValueError(f"{cls.TYPE}: {e}") from None

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

    def accept(self):
        problems = self.problems()
        if problems:
            QtWidgets.QMessageBox.warning(self, "Region", "\n".join(problems))
            return
        super().accept()


# ---------------------------------------------------------------------- a support or load
def field_unit(measures, units):
    from databases import UNIT_SYSTEMS  # type: ignore
    if measures is None or units not in UNIT_SYSTEMS:
        return ""
    s = UNIT_SYSTEMS[units]
    return f"{s['length']}^2" if measures == "area" else s.get(measures, "")


class BCDialog(QtWidgets.QDialog):
    """A support or load: its kind, its region, and the kind's values (BCKind.fields)."""

    def __init__(self, parent, project, bc=None):
        from project import BC_KINDS  # type: ignore
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.setWindowTitle("Support or load")
        self.resize(560, 300)
        self.project, self.old, self.kinds = project, bc, BC_KINDS
        layout = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QFormLayout()
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(list(BC_KINDS))
        self.on = QtWidgets.QComboBox()
        self.on.addItems(list(project.selections))
        top.addRow("Kind", self.kind)
        top.addRow("Region", self.on)
        layout.addLayout(top)
        self.doc = QtWidgets.QLabel()
        self.doc.setStyleSheet(GREY)
        self.doc.setWordWrap(True)
        layout.addWidget(self.doc)
        self.form = QtWidgets.QFormLayout()
        layout.addLayout(self.form)
        layout.addStretch()
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.inputs = {}
        self.kind.currentTextChanged.connect(lambda k: self._fill(k, None))
        if bc:
            self.kind.setCurrentText(bc["kind"])
            self.on.setCurrentText(bc["on"])
            self._fill(bc["kind"], bc)
        else:
            self._fill(self.kind.currentText(), None)

    def _fill(self, kind, bc):
        while self.form.rowCount():
            self.form.removeRow(0)
        self.doc.setText(self.kinds[kind].doc)
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
                widget = (QtWidgets.QCheckBox("bbox"), NumberEdit(0.0 if value in (None, "bbox") else value))
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
            label = QtWidgets.QLabel(name)
            label.setToolTip(doc)
            self.form.addRow(label, row)
            self.inputs[name] = (typ, widget)

    def bc(self):
        """The support or load as the dialog shows it, each unchanged value exactly as it was."""
        kind = self.kind.currentText()
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
            out.append("no region: add one first")
        for name, (typ, widget) in self.inputs.items():
            boxes = widget if typ == "vector" else [widget[1]] if typ == "area" and not widget[0].isChecked() \
                else [widget] if typ == "number" else []
            if any(not b.ok() for b in boxes):
                out.append(f"{name} is not a number")
            if typ == "dofs" and not any(b.isChecked() for b in widget.values()):
                out.append(f"{name}: tick at least one of x, y, z")
        if not out:
            out += self.kinds[self.kind.currentText()].check(self.bc())
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
    """Regions, supports and loads, keep solid, and the preview. Changes go into the project at once."""

    def __init__(self, window):
        super().__init__(window)
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setWindowTitle("Supports and loads")
        self.resize(760, 620)
        self.window = window
        self.counts = {}                                            # region -> nodes, after a preview
        layout = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        for text, slot in (("Pick with the mouse...", self.pick), ("Preview", self.preview),
                           ("Clear preview", self.clear_preview)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            top.addWidget(button)
        top.addStretch()
        layout.addLayout(top)

        layout.addWidget(QtWidgets.QLabel("<b>Regions</b>"))
        self.regions = QtWidgets.QTableWidget(0, 4)
        self.regions.setHorizontalHeaderLabels(["Name", "Definition", "Nodes", "Keep solid"])
        self.regions.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.regions.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.regions.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.regions.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.regions.doubleClicked.connect(lambda _: self.edit_region())
        layout.addWidget(self.regions, stretch=1)
        layout.addLayout(self._buttons((("Add...", self.add_region), ("Edit...", self.edit_region),
                                        ("Rename...", self.rename_region), ("Delete", self.delete_region))))

        layout.addWidget(QtWidgets.QLabel("<b>Supports and loads</b>"))
        self.bcs = QtWidgets.QTableWidget(0, 3)
        self.bcs.setHorizontalHeaderLabels(["Kind", "Region", "Values"])
        self.bcs.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.bcs.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.bcs.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.bcs.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.bcs.doubleClicked.connect(lambda _: self.edit_bc())
        layout.addWidget(self.bcs, stretch=1)
        layout.addLayout(self._buttons((("Add...", self.add_bc), ("Edit...", self.edit_bc),
                                        ("Delete", self.delete_bc))))

        self.report = QtWidgets.QLabel()
        self.report.setWordWrap(True)
        layout.addWidget(self.report)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.close)
        layout.addWidget(close, alignment=QtCore.Qt.AlignRight)
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
            self.regions.setItem(r, 1, QtWidgets.QTableWidgetItem(repr(p.selections[name])))
            self.regions.setItem(r, 2, QtWidgets.QTableWidgetItem(str(self.counts.get(name, "-"))))
            keep = QtWidgets.QCheckBox()
            keep.setChecked(name in p.keep_solid_names)
            keep.toggled.connect(lambda on, n=name: self.set_keep_solid(n, on))
            self.regions.setCellWidget(r, 3, keep)
        self.regions.resizeColumnToContents(0)
        self.bcs.setRowCount(len(p.bcs))
        for r, bc in enumerate(p.bcs):
            values = ", ".join(f"{k}={v}" for k, v in bc.items() if k not in ("kind", "on"))
            for c, text in enumerate((bc["kind"], bc["on"], values)):
                self.bcs.setItem(r, c, QtWidgets.QTableWidgetItem(text))
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
        if not self.project.selections:
            QtWidgets.QMessageBox.information(self, "Support or load", "Add a region first.")
            return None
        dialog = BCDialog(self, self.project)
        if dialog.exec_():
            self.project.bcs.append(dialog.bc())
            self.after_change()
        return dialog

    def edit_bc(self, index=None):
        index = self._selected(self.bcs) if index is None else index
        if index is None:
            return None
        dialog = BCDialog(self, self.project, self.project.bcs[index])
        if dialog.exec_():
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
        """PyTO-AD's Structural Loads window; each Apply in it is adopted into the project."""
        self.window.open_structural_loads_window()

    def after_change(self):
        self.refresh()
        if self.window.preview_on:
            self.preview()

    def preview_one(self, selection):
        """A region being edited, on the run's mesh: its node count, drawn in orange."""
        mesh = self.window.preview_mesh()
        nodes = selection.evaluate(mesh, "nodes", self.project.base_dir)
        self.window.draw_points("mmto_region", np.asarray(mesh.node_xyz)[nodes], OTHER_COLOUR)
        return f"{len(nodes)} nodes on the run's mesh ({mesh.num_elems} elements)."

    def preview(self):
        """Every region, support and load on the run's mesh, the node counts, and the rigid-body check."""
        self.report.setText("")
        try:
            lines, self.counts = self.window.draw_supports_and_loads()
        except Exception as e:                                       # noqa: BLE001
            lines = [f"<span style='color:#b00000'>Error: {e}</span>"]
        self.report.setText("<br>".join(lines))
        self.refresh()

    def clear_preview(self):
        """The preview, and any faces or nodes picked but not applied."""
        self.window.clear_preview()
        self.window.clear_picks()
        self.counts = {}
        self.report.setText("")
        self.refresh()
