"""
formulation_panel.py
====================
The Formulation panel of the LSR MMTO window: the objective and the constraints.
OK and Cancel work as in every panel (see materials_panel.py).
"""
from PyQt5 import QtCore, QtGui, QtWidgets

from .panel_widgets import GREY, Info, NumberEdit, database_choice, ok_cancel, same, swatch


TAU_DOC = ("How sharply each element counts as its nearest material when a material's volume is measured: "
           "small is sharp, large is blurred. One value for every material volume fraction constraint.")


def bound_unit(name, units):
    """The unit a bound on the quantity is in, in the project's unit system ('' when none declared)."""
    from lsr_mmto.materials.databases import UNIT_SYSTEMS
    if name in ("VolumeFraction", "MaterialVolumeFraction"):
        return "fraction of the design domain"
    if name == "FailureFactorPNorm":
        return "no units"
    if units not in UNIT_SYSTEMS:
        return "no unit system declared"
    s = UNIT_SYSTEMS[units]
    return {"Mass": s["mass"], "Cost": "USD", "Compliance": f"{s['force']}*{s['length']}",
            "StressPNorm": s["stress"]}.get(name, "")


def database_columns(project):
    """The database's columns, the materials' labels and colours; None for each it cannot read."""
    from lsr_mmto.materials.databases import check_database, database_attributes, material_display
    xlsx, error = database_choice(project)
    if error or check_database(xlsx)[0]:
        return None, None, None
    colours, labels = material_display(xlsx)
    return database_attributes(xlsx), labels, colours


class FormulationPanel(QtWidgets.QDialog):
    """The objective (minimized) and the constraints (quantity <= bound). A quantity the database
    cannot give is greyed out with the reason."""

    def __init__(self, parent, project):
        from lsr_mmto.formulation.quantities import QUANTITIES
        super().__init__(parent)
        self.setWindowTitle("Formulation")
        self.resize(760, 420)
        self.original = project.formulation
        self.units = project.units
        self.quantities = {q.name: q for q in QUANTITIES}
        self.columns, self.labels, self.colours = database_columns(project)
        layout = QtWidgets.QVBoxLayout(self)

        layout.addWidget(QtWidgets.QLabel("<b>Minimize</b>"))
        self.objective = self._quantity_combo("objective")
        obj = self.original.get("objective")
        self._set_quantity(self.objective, obj and obj.get("expression"))
        layout.addWidget(self.objective)

        layout.addWidget(QtWidgets.QLabel("<b>Constraints</b>"))
        self.tau_box = QtWidgets.QWidget()                      # shown only with a material volume fraction
        tau_row = QtWidgets.QHBoxLayout(self.tau_box)
        tau_row.setContentsMargins(0, 0, 0, 0)
        self.tau = NumberEdit(self._stored_tau())
        self.tau.setFixedWidth(80)
        for widget in (QtWidgets.QLabel("Material sharpness (tau)"), self.tau, Info(TAU_DOC)):
            tau_row.addWidget(widget)
        tau_row.addStretch()
        layout.addWidget(self.tau_box)
        self.rows_box = QtWidgets.QVBoxLayout()
        layout.addLayout(self.rows_box)
        self.rows = []
        for c in self.original.get("constraints", []):
            self.add_row(c)
        add = QtWidgets.QPushButton("Add constraint")
        add.clicked.connect(lambda: self.add_row(None))
        layout.addWidget(add, alignment=QtCore.Qt.AlignLeft)
        if self.columns is None:
            note = QtWidgets.QLabel("The database cannot be read, so no quantity is greyed out for missing columns.")
            note.setStyleSheet(GREY)
            layout.addWidget(note)
        layout.addStretch()
        self._show_tau()
        ok_cancel(self, layout)

    def _stored_tau(self):
        """The tau of the project's material volume fraction constraints (the first, if they differ)."""
        from lsr_mmto.formulation.quantities import DEFAULT_TAU
        from lsr_mmto.formulation.formulation_text import parse_expression
        for c in self.original.get("constraints", []):
            try:
                name, _, kwargs = parse_expression(c.get("expression"))
            except ValueError:
                continue
            if name == "MaterialVolumeFraction":
                return kwargs.get("tau", DEFAULT_TAU)
        return DEFAULT_TAU

    def _show_tau(self):
        self.tau_box.setVisible(any(e["widgets"]["quantity"].currentData() == "MaterialVolumeFraction"
                                    for e in self.rows))

    def reason(self, name):
        """Why the quantity cannot be used with this database, or None."""
        if self.columns is None:
            return None
        missing = [c for c in self.quantities[name].needs if c not in self.columns]
        if missing:
            return f"needs a {' and a '.join(missing)} column in the database"
        return None

    def _quantity_combo(self, role):
        combo = QtWidgets.QComboBox()
        model = QtGui.QStandardItemModel()
        for name, q in self.quantities.items():
            if not (q.objective if role == "objective" else q.constraint):
                continue
            item = QtGui.QStandardItem(q.shown)
            item.setData(name, QtCore.Qt.UserRole)
            why = self.reason(name)
            if why:
                item.setEnabled(False)
                item.setToolTip(f"Not available: {why}")
            else:
                item.setToolTip(q.help)
            model.appendRow(item)
        combo.setModel(model)
        combo.setCurrentIndex(-1)
        return combo

    def _set_quantity(self, combo, expression):
        from lsr_mmto.formulation.formulation_text import parse_expression
        if not expression:
            combo.setCurrentIndex(-1)
            return
        try:
            name = parse_expression(expression)[0]
        except ValueError:
            name = None
        combo.setCurrentIndex(combo.findData(name) if name else -1)

    def add_row(self, constraint):
        """A constraint row: quantity, material (MaterialVolumeFraction only), <=, bound, unit."""
        from lsr_mmto.formulation.formulation_text import parse_expression
        row = QtWidgets.QHBoxLayout()
        widgets = {"quantity": self._quantity_combo("constraint"), "material": QtWidgets.QComboBox(),
                   "bound": NumberEdit(0.0), "unit": QtWidgets.QLabel(),
                   "why": QtWidgets.QLabel(), "remove": QtWidgets.QPushButton("Remove")}
        n = len(self.labels) if self.labels else 0
        for i in range(n):
            widgets["material"].addItem(swatch(self.colours[i]), self.labels[i], i)
        widgets["bound"].setFixedWidth(150)
        widgets["unit"].setStyleSheet(GREY)
        widgets["why"].setStyleSheet("color: #b00000;")
        for key in ("quantity", "material", "<=", "bound", "unit", "why"):
            row.addWidget(QtWidgets.QLabel("<=") if key == "<=" else widgets[key])
        row.addStretch()
        row.addWidget(widgets["remove"])
        entry = {"widgets": widgets, "layout": row, "original": constraint}
        if constraint:
            self._set_quantity(widgets["quantity"], constraint.get("expression"))
            widgets["bound"].set_value(constraint.get("bound") or 0.0)
            try:
                name, args, kwargs = parse_expression(constraint["expression"])
                if name == "MaterialVolumeFraction" and args:
                    i = int(args[0])
                    if i >= widgets["material"].count():
                        widgets["material"].addItem(f"material {i}", i)
                    widgets["material"].setCurrentIndex(widgets["material"].findData(i))
            except ValueError:
                pass
        widgets["quantity"].currentIndexChanged.connect(lambda _=None, e=entry: self._update_row(e))
        widgets["remove"].clicked.connect(lambda _=False, e=entry: self.remove_row(e))
        self.rows.append(entry)
        self.rows_box.addLayout(row)
        self._update_row(entry)
        return entry

    def _update_row(self, entry):
        w = entry["widgets"]
        name = w["quantity"].currentData()
        w["material"].setVisible(name == "MaterialVolumeFraction")
        unit = bound_unit(name, self.units) if name else ""
        w["unit"].setText("" if unit in ("fraction of the design domain", "no units") else unit)
        w["why"].setText(self.reason(name) or "" if name else "")
        if hasattr(self, "tau_box"):
            self._show_tau()

    def remove_row(self, entry):
        self.rows.remove(entry)
        while entry["layout"].count():
            widget = entry["layout"].takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        self.rows_box.removeItem(entry["layout"])
        self._show_tau()

    def problems(self):
        """What keeps OK from closing: an unfinished row or a number that is not one."""
        out = []
        for k, e in enumerate(self.rows, 1):
            w = e["widgets"]
            if not w["quantity"].currentData():
                out.append(f"constraint {k}: choose a quantity")
            if not w["bound"].ok():
                out.append(f"constraint {k}: the bound is not a number")
        if self.tau_box.isVisibleTo(self) and not (self.tau.ok() and self.tau.value() > 0):
            out.append("tau must be a number greater than 0")
        return out

    def accept(self):
        problems = self.problems()
        if problems:
            QtWidgets.QMessageBox.warning(self, "Formulation", "\n".join(problems))
            return
        super().accept()

    def formulation(self):
        """The formulation as the panel shows it, each unchanged entry exactly as it was."""
        obj = self.original.get("objective")
        name = self.objective.currentData()
        if not name:
            objective = None
        elif obj and self._same_expression(obj.get("expression"), f"{name}()"):
            objective = obj
        else:
            objective = {"expression": f"{name}()", "sense": "minimize"}
        constraints = []
        for e in self.rows:
            w, old = e["widgets"], e["original"] or {}
            name = w["quantity"].currentData()
            if name == "MaterialVolumeFraction":
                expression = f"MaterialVolumeFraction({w['material'].currentData() or 0}, tau={self.tau.value():g})"
            else:
                expression = f"{name}()"
            new = {"expression": expression, "op": "<=", "bound": w["bound"].value()}
            if old and self._same_expression(old.get("expression"), expression):
                new["expression"] = old["expression"]
            if old and same(old.get("bound"), new["bound"]):
                new["bound"] = old["bound"]
            constraints.append(dict(old, **new) if old else new)
        return {"objective": objective, "constraints": constraints}

    @staticmethod
    def _same_expression(a, b):
        from lsr_mmto.formulation.formulation_text import parse_expression
        try:
            return parse_expression(a) == parse_expression(b)
        except ValueError:
            return False

    def apply(self, project):
        f = self.formulation()
        if f == project.formulation:
            return []
        project.formulation = f
        return formulation_lines(f, project)


def formulation_lines(f, project):
    """The formulation in words, for the message panel: Minimize Stress (p-norm); Mass <= 0.4 t; ..."""
    from lsr_mmto.formulation.formulation_text import parse_expression
    _, labels, _ = database_columns(project)

    def words(expression):
        try:
            name, args, _ = parse_expression(expression)
        except ValueError:
            return expression
        from lsr_mmto.formulation.quantities import QUANTITIES
        shown = next((q.shown for q in QUANTITIES if q.name == name), name)
        if name == "MaterialVolumeFraction" and args:
            i = int(args[0])
            shown += f" of {labels[i] if labels and i < len(labels) else f'material {i}'}"
        return shown

    obj = f.get("objective")
    lines = ["Minimize " + (words(obj["expression"]) if obj else "(nothing chosen)")]
    for c in f.get("constraints", []):
        try:
            unit = bound_unit(parse_expression(c["expression"])[0], project.units)
        except ValueError:
            unit = ""
        unit = "" if unit in ("no units", "fraction of the design domain", "no unit system declared") else f" {unit}"
        lines.append(f"Constraint: {words(c['expression'])} <= {c['bound']:g}{unit}")
    return lines
