"""
panels.py
=========
The Materials, Formulation and Settings panels of the LSR MMTO window. Each is a dialog with
OK and Cancel working on a copy of the project: OK hands back the edited parts (apply), Cancel
leaves the project as it was. A value left as it was is handed back exactly as it was, so a
panel closed with OK and no edits changes nothing in the file.
"""
import math
import os
import subprocess
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

GREY = "color: #666666;"


# ---------------------------------------------------------------------- small widgets
class NumberEdit(QtWidgets.QLineEdit):
    """A number typed as text, shown to 12 significant digits: 1e-07, 0.12345, and 0.0035 for
    the 0.0035000000000000005 that 0.40 * 0.875 * 0.01 gives. A field left as shown hands back the
    stored number exactly; only a typed change replaces it. (PyTO-AD's ScientificDoubleSpinBox
    shows three digits, 0.12345 as 0.12, and hands back what it shows.)"""

    def __init__(self, value=0.0, integer=False, parent=None):
        super().__init__(parent)
        self.integer = integer
        self.set_value(value)
        self.textChanged.connect(lambda _: self.setStyleSheet("" if self.ok() else "background: #ffd6d6;"))

    def set_value(self, value):
        self._exact = int(value) if self.integer else float(value)
        self._shown = str(self._exact) if self.integer else format(self._exact, ".12g")
        self.setText(self._shown)

    def ok(self):
        try:
            v = float(self.text())
        except ValueError:
            return False
        return math.isfinite(v) and (not self.integer or v == int(v))

    def value(self):
        if self.text() == self._shown:
            return self._exact
        v = float(self.text())
        return int(v) if self.integer else v


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


def _database_errors(project):
    from lsr_mmto.materials.databases import check_database
    xlsx, _ = project.database_paths()
    return check_database(xlsx)[0]


def effective_text(d_eff, n_z):
    """d_eff with enough digits that rounding it up visibly gives n_z (1.004, not 1.00, for 2)."""
    for digits in (2, 3, 4, 6):
        text = f"{d_eff:.{digits}f}"
        if max(1, math.ceil(float(text))) == n_z:
            return text
    return f"{d_eff:.6g}"


def latent_help(d_eff, n_z, ratios):
    """Where the recommended latent dimension comes from, in words: the share of the variation in
    each direction the materials differ in, and the effective dimension counted from them."""
    lines = ["How the materials differ. With each property scaled to 0 to 1, the ways they differ are split "
             "into independent directions, largest first:"]
    total = 0.0
    for i, r in enumerate(ratios):
        total += float(r)
        lines.append(f"    direction {i + 1}: {100 * r:.1f}% of the variation (in all, {100 * total:.1f}%)")
    lines.append(f"The effective dimension, {effective_text(d_eff, n_z)}, counts how many of these directions "
                 f"the materials really differ in. The recommended latent dimension is the smallest whole "
                 f"number at or above it: {n_z}.")
    return "\n".join(lines)


def database_choice(project):
    """(xlsx path or None, error or None) of the project's database."""
    if project.database is None:
        return None, "no database chosen"
    xlsx, _ = project.database_paths()
    return (xlsx, None) if os.path.isfile(xlsx) else (xlsx, f"{xlsx} does not exist")


# ---------------------------------------------------------------------- Materials
class MaterialsPanel(QtWidgets.QDialog):
    """The material database: which one, its materials in their colours and labels, its check,
    and the latent dimension recommended for it. Choosing another database resets the VAE
    settings to the new one's (its trained values, or the recommended latent dimension)."""

    def __init__(self, parent, project):
        from lsr_mmto.materials.databases import DATABASE_VAE
        super().__init__(parent)
        self.setWindowTitle("Materials")
        self.resize(820, 520)
        self.project = project.copy()
        self.original = project.database
        layout = QtWidgets.QVBoxLayout(self)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Database:"))
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(list(DATABASE_VAE))
        if project.database is not None and project.database not in DATABASE_VAE:
            self.combo.addItem(project.database)
        self.combo.setCurrentIndex(-1 if project.database is None else self.combo.findText(project.database))
        self.combo.currentTextChanged.connect(self._choose)
        row.addWidget(self.combo, stretch=1)
        for text, slot in (("New database...", self.new_database), ("Edit...", self.edit_database),
                           ("Browse...", self._browse), ("Open in Excel", self._open_in_excel),
                           ("Check again", self._refresh)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
        layout.addLayout(row)

        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table, stretch=1)
        self.report = QtWidgets.QTextEdit()
        self.report.setReadOnly(True)
        self.report.setFixedHeight(64)
        below = QtWidgets.QHBoxLayout()
        below.addWidget(self.report, stretch=1)
        self.latent_info = Info("")                                 # how the latent dimension is recommended
        self.latent_info.setVisible(False)
        below.addWidget(self.latent_info, alignment=QtCore.Qt.AlignTop)
        layout.addLayout(below)
        ok_cancel(self, layout)
        self._refresh()

    def _choose(self, name):
        if name:
            self.project.database = name
            self._refresh()

    def _browse(self):
        from lsr_mmto.materials.databases import DATABASE_DIR
        from .pyto_ad import FILE_DIALOG_OPTIONS
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Material database", DATABASE_DIR,
                                                        "Excel files (*.xlsx)", options=FILE_DIALOG_OPTIONS)
        if path:
            self.set_database(path)

    def _own_folder(self, what):
        """The project's materials folder, or None after saying the project must be saved first."""
        window = self.parent()
        if hasattr(window, "own_folder") and not window.own_folder():
            QtWidgets.QMessageBox.information(self, what, "Save the project first (File > Save), so that the "
                                                          "database can be kept in its folder.")
            return None
        return window.database_folder() if hasattr(window, "database_folder") else self.project.base_dir

    def new_database(self):
        """A database made in the editor (database_editor.py), saved in the project's folder and then used
        by this project."""
        from .database_editor import DatabaseEditor
        folder = self._own_folder("New database")
        if folder is None:
            return None
        editor = DatabaseEditor(self, folder, unit_system=self.project.units)
        if not editor.exec_():
            return None
        path = editor.saved_path or editor.save()
        self.set_database(path)
        self.report.setText(f"{os.path.basename(path)} saved in the project's folder.")
        return path

    def edit_database(self):
        """The database in the editor: one of the project's own is changed in place; one that comes with
        LSR MMTO is saved as a copy in the project's folder, which the project then uses."""
        from lsr_mmto.materials.databases import in_shipped_folder
        from .database_editor import DatabaseEditor
        xlsx, error = database_choice(self.project)
        if error:
            self.report.setText(error)
            return None
        shipped = in_shipped_folder(xlsx)
        folder = self._own_folder("Database") if shipped else os.path.dirname(os.path.abspath(xlsx))
        if folder is None:
            return None
        editor = DatabaseEditor(self, folder, path=xlsx, unit_system=self.project.units)
        if not editor.exec_():
            return None
        path = editor.saved_path or editor.save()
        self.set_database(path)
        self._refresh()
        note = (f"Saved as {os.path.basename(path)} in the project's folder." if shipped else
                f"{os.path.basename(path)} saved.")
        if editor.network_set_aside:
            note += (f" Its numbers changed, so the network trained on the old ones was set aside as "
                     f"{os.path.basename(editor.network_set_aside)}; the next run trains a new one.")
        self.report.setText(note + "\n" + self.report.toPlainText())
        return path

    def set_database(self, path):
        """A database by its file: a shipped one by name, any other by its path relative to the project."""
        from lsr_mmto.materials.databases import DATABASE_DIR, DATABASE_VAE
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem in DATABASE_VAE and os.path.samefile(os.path.dirname(os.path.abspath(path)),
                                                     os.path.abspath(DATABASE_DIR)):
            name = stem
        else:
            try:
                name = os.path.relpath(os.path.abspath(path), self.project.base_dir).replace("\\", "/")
            except ValueError:                                      # another drive on Windows
                name = os.path.abspath(path)
        if self.combo.findText(name) < 0:
            self.combo.addItem(name)
        self.combo.setCurrentText(name)

    def _open_in_excel(self):
        xlsx, error = database_choice(self.project)
        if error:
            self.report.setText(error)
            return
        if sys.platform.startswith("win"):
            os.startfile(os.path.abspath(xlsx))                     # noqa: S606
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", xlsx])

    def _refresh(self):
        """Read the database again: the table, the check, the recommended latent dimension."""
        from lsr_mmto.materials.databases import check_database, database_table, latent_dim_breakdown, material_display
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.latent_info.setVisible(False)
        xlsx, error = database_choice(self.project)
        if error:
            self.report.setText(error)
            return
        errors, warnings = check_database(xlsx)
        lines = [f"Error: {e}" for e in errors] + [f"Warning: {w}" for w in warnings]
        try:
            names, attributes, units, values = database_table(xlsx)
            colours, labels = material_display(xlsx) if not errors else ([None] * len(names), names)
        except Exception as e:                                      # noqa: BLE001
            lines.append(f"The table could not be read: {e}")
            self.report.setText("\n".join(lines))
            return
        from lsr_mmto.materials.databases import SHOWN
        heads = ["Label", "Material"] + [SHOWN.get(a, a) + ("" if u is None else f" [{u}]")
                                         for a, u in zip(attributes, units)]
        self.table.setColumnCount(len(heads))
        self.table.setHorizontalHeaderLabels(heads)
        self.table.setRowCount(len(names))
        for i, name in enumerate(names):
            item = QtWidgets.QTableWidgetItem(labels[i])
            if colours[i]:
                item.setIcon(swatch(colours[i]))
            self.table.setItem(i, 0, item)
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(name))
            for j in range(values.shape[1]):
                self.table.setItem(i, 2 + j, QtWidgets.QTableWidgetItem(format(values[i, j], ".6g")))
        self.table.verticalHeader().setVisible(False)
        self.table.setColumnHidden(1, list(labels) == list(names))      # Material: only when the labels differ
        self.table.resizeColumnsToContents()
        if not errors:
            d_eff, n_z, ratios = latent_dim_breakdown(xlsx)
            lines.insert(0, f"{len(names)} materials; latent dimension {n_z} recommended (effective dimension "
                            f"{effective_text(d_eff, n_z)}).")
            self.latent_info.setToolTip(latent_help(d_eff, n_z, ratios))
            self.latent_info.setVisible(True)
        self.report.setText("\n".join(lines))

    def apply(self, project):
        """The database, and on a change the VAE settings of the new one. Returns lines saying what changed."""
        from lsr_mmto.problem.project import VAE_PARAMS
        if self.project.database == self.original:
            return []
        q = self.project.copy()
        q.settings["vae"] = {"use_pretrained_vae": project.settings["vae"].get("use_pretrained_vae", True)}
        q.complete_settings()
        before = {k: project.settings["vae"].get(k) for k in VAE_PARAMS + ("saveNet",)}
        project.database = q.database
        project.settings["vae"] = q.settings["vae"]
        changed = [f"{LABELS.get(k, k)} {before[k]} -> {q.settings['vae'][k]}" for k in before
                   if not same(before[k], q.settings["vae"][k])]
        return [f"Database: {q.database}"] + ([f"VAE settings for it: {', '.join(changed)}"] if changed else [])


# ---------------------------------------------------------------------- Formulation
TAU_DOC = ("How sharply each element counts as its nearest material when a material's volume is measured: "
           "small is sharp, large is blurred. One value for every material volume fraction constraint.")
# the quantities as the panel shows them, and what each is
QUANTITY_NAMES = {"Compliance": "Compliance", "Mass": "Mass", "Cost": "Cost", "StressPNorm": "Stress (p-norm)",
                  "FailureFactorPNorm": "Failure factor (p-norm)", "VolumeFraction": "Volume fraction",
                  "MaterialVolumeFraction": "Material volume fraction"}
QUANTITY_HELP = {
    "Compliance": "How much the part gives under the loads. Lower is stiffer.",
    "Mass": "The mass of the part, from each material's density.",
    "Cost": "The cost of the part, from each material's density and cost per mass.",
    "StressPNorm": "One value close to the highest von Mises stress in the part.",
    "FailureFactorPNorm": "One value close to the highest stress / yield strength in the part. Above 1 means "
                          "some element yields.",
    "VolumeFraction": "The share of the design space filled with material, 0 to 1.",
    "MaterialVolumeFraction": "The share of the design space filled with one chosen material, 0 to 1.",
}


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
        from lsr_mmto.physics.structural import QUANTITIES
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
        from lsr_mmto.physics.structural import DEFAULT_TAU
        from lsr_mmto.problem.project import parse_expression
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
            if (q.objective if role == "objective" else q.constraint) is None:
                continue
            item = QtGui.QStandardItem(QUANTITY_NAMES.get(name, name))
            item.setData(name, QtCore.Qt.UserRole)
            why = self.reason(name)
            if why:
                item.setEnabled(False)
                item.setToolTip(f"Not available: {why}")
            else:
                item.setToolTip(QUANTITY_HELP.get(name, ""))
            model.appendRow(item)
        combo.setModel(model)
        combo.setCurrentIndex(-1)
        return combo

    def _set_quantity(self, combo, expression):
        from lsr_mmto.problem.project import parse_expression
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
        from lsr_mmto.problem.project import parse_expression
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
        from lsr_mmto.problem.project import parse_expression
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
    from lsr_mmto.problem.project import parse_expression
    _, labels, _ = database_columns(project)

    def words(expression):
        try:
            name, args, _ = parse_expression(expression)
        except ValueError:
            return expression
        shown = QUANTITY_NAMES.get(name, name)
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


# ---------------------------------------------------------------------- Settings
LABELS = {
    "n_dof": "Mesh dofs", "single_layer": "Single element layer (2D)",
    "maxIterations": "Max iterations", "relative_filter_radius": "Density filter radius",
    "material_filter_radius": "Latent smoothing radius", "pnorm_exponent": "p-norm exponent",
    "beta_schedule_scheme": "Projection sharpness schedule",
    "timeLimit": "Time limit [s]", "rel_conv_tol": "Convergence tolerance",
    "use_heaviside_projection": "Heaviside projection", "eta_proj": "Projection threshold",
    "use_penalization": "Distance penalty", "gamma_init": "Starting weight", "gamma_max": "Largest weight",
    "gamma_factor": "Weight growth per iteration", "penalty_start_iter": "First iteration",
    "distance_penalty_disable_after_iter": "Off after iteration",
    "use_dynamic_gamma_scheduling": "Start the ramp when the objective plateaus",
    "use_z_smoothing": "Latent smoothing", "z_smoothing_weight": "Weight",
    "z_smoothing_disable_after_iter": "Off after iteration",
    "use_continuation": "SIMP continuation", "use_constant_poissons_ratio": "Constant Poisson's ratio",
    "constant_poissons_ratio": "Poisson's ratio value",
    "z0_init_method": "Starting latent point", "snap_to_real_material": "Snap to real materials",
    "binarize_topology": "Binarize the topology",
    "use_pretrained_vae": "Use the trained network", "saveNet": "VAE file",
    "network_error_warning": "Warn above network error [%]", "latentDim": "Latent dimension",
    "learningRate": "Learning rate", "vae_hiddenDim": "Hidden layer width",
    "maxAttributeErrorPercent": "Stop at attribute error [%]", "klFactor": "KL weight",
    "numEpochs": "Max epochs",
}
# what each setting does, in plain words: shown by its (i)
HELP = {
    "n_dof": "How fine the mesh is: the number of unknowns it is built for. More is finer, and slower.",
    "single_layer": "One element through the thickness, for a flat (2D) part.",
    "maxIterations": "The optimization stops after this many iterations.",
    "relative_filter_radius": "The smallest feature the design may have, in element sizes. Larger gives smoother, "
                              "simpler designs.",
    "material_filter_radius": "How far material choices are smoothed between neighbouring elements, in element "
                              "sizes. Ticked: the same as the density filter radius.",
    "pnorm_exponent": "How closely the single stress value follows the highest stress in the part. Larger is "
                      "closer, but harder to optimize.",
    "beta_schedule_scheme": "How fast the design is pushed to clear solid and empty: PAPER sharpens up to 32, "
                            "PAPER_BM1 up to 16.",
    "timeLimit": "The optimization stops after this many seconds.",
    "rel_conv_tol": "The optimization stops early once the design changes by less than this, relatively, "
                    "between iterations.",
    "use_heaviside_projection": "Pushes each element towards fully solid or fully empty.",
    "eta_proj": "The density at which that push splits solid from empty.",
    "use_penalization": "Pushes each element's material towards one of the real materials in the database.",
    "gamma_init": "How strong that push is at the start.",
    "gamma_max": "How strong it can get.",
    "gamma_factor": "How much stronger it gets each iteration.",
    "penalty_start_iter": "The iteration it starts at.",
    "distance_penalty_disable_after_iter": "The iteration it stops at.",
    "use_dynamic_gamma_scheduling": "Start the push when the design stops improving, instead of at a set "
                                    "iteration.",
    "use_z_smoothing": "Discourages neighbouring elements from flickering between materials.",
    "z_smoothing_weight": "How strong that smoothing is.",
    "z_smoothing_disable_after_iter": "The iteration it stops at.",
    "use_continuation": "Raises the SIMP penalty step by step during the run.",
    "use_constant_poissons_ratio": "One Poisson's ratio for every material, instead of each material's own.",
    "constant_poissons_ratio": "The Poisson's ratio used when Constant Poisson's ratio is ticked.",
    "z0_init_method": "The material every element starts with: ORIGIN (the middle of the material map), the "
                      "LIGHTEST or HEAVIEST material, or UNIFORM (random).",
    "snap_to_real_material": "In the final design, give each element the nearest real material.",
    "binarize_topology": "In the final design, make each element fully solid or empty, keeping the volume.",
    "use_pretrained_vae": "Use the network already trained for this database. Unticked, the next run trains a "
                          "new one.",
    "saveNet": "The trained network: always the .nt file next to the database, with the same name.",
    "network_error_warning": "Before a run, you are warned if the network reproduces any material property "
                             "less closely than this, in % of the property's range in the database. You can "
                             "still run.",
    "latentDim": "How many numbers describe one material in the material map.",
    "learningRate": "The step size of the network's training.",
    "vae_hiddenDim": "The size of the network's inner layers.",
    "maxAttributeErrorPercent": "Training stops once the network reproduces every material property within "
                                "this percent.",
    "klFactor": "How strongly training keeps the material map compact and smooth.",
    "numEpochs": "Training stops after this many passes at the latest.",
}
# (tab, group, [(section, key), ...]); section "mesh" is Project.n_dof and Project.single_layer
LAYOUT = [
    ("Optimization", "Main", [("mesh", "n_dof"), ("mesh", "single_layer"), ("common", "maxIterations"),
                              ("common", "relative_filter_radius"), ("common", "material_filter_radius"),
                              ("common", "pnorm_exponent"), ("common", "beta_schedule_scheme")]),
    ("Optimization", "Run length", [("advanced", "timeLimit"), ("advanced", "rel_conv_tol")]),
    ("Optimization", "Projection", [("advanced", "use_heaviside_projection"), ("advanced", "eta_proj")]),
    ("Optimization", "Distance penalty", [("advanced", k) for k in (
        "use_penalization", "gamma_init", "gamma_max", "gamma_factor", "penalty_start_iter",
        "distance_penalty_disable_after_iter", "use_dynamic_gamma_scheduling")]),
    ("Optimization", "Latent smoothing", [("advanced", k) for k in (
        "use_z_smoothing", "z_smoothing_weight", "z_smoothing_disable_after_iter")]),
    ("Optimization", "Material model", [("advanced", k) for k in (
        "use_continuation", "use_constant_poissons_ratio", "constant_poissons_ratio")]),
    ("Optimization", "Start and reported design", [("advanced", k) for k in (
        "z0_init_method", "snap_to_real_material", "binarize_topology")]),
    ("VAE", "VAE", [("vae", k) for k in ("use_pretrained_vae", "saveNet", "network_error_warning", "latentDim",
                                         "learningRate", "vae_hiddenDim", "maxAttributeErrorPercent", "klFactor",
                                         "numEpochs")]),
]
# settings that may be None, and what None means
OPTIONAL = {"material_filter_radius": "same as density filter",
            "saveNet": "next to the database"}
VAE_KEYS = ("latentDim", "learningRate", "vae_hiddenDim", "maxAttributeErrorPercent", "klFactor", "numEpochs")
VAE_NOTE = ("A database's network is the .nt file next to its .xlsx, with the same name (my_db.xlsx and "
            "my_db.nt). If there is none, the first run trains one. A database that comes with LSR MMTO can't be "
            "retrained: copy its .xlsx to a folder of your own first.")
INTEGERS = {"n_dof", "maxIterations", "timeLimit", "penalty_start_iter", "distance_penalty_disable_after_iter",
            "z_smoothing_disable_after_iter", "latentDim", "vae_hiddenDim", "numEpochs"}


def shown_setting(key, value):
    """A setting's value in words: on/off, a number, or what an empty one means."""
    if value is None:
        return OPTIONAL.get(key, "none")
    if isinstance(value, bool):
        return "on" if value else "off"
    return format(value, "g") if isinstance(value, (int, float)) else str(value)


def setting_choices():
    """The settings with a fixed list of values."""
    from lsr_mmto.optimize.run import BetaScheduleScheme, Z0InitMethod
    return {"beta_schedule_scheme": list(BetaScheduleScheme.__members__),
            "z0_init_method": list(Z0InitMethod.__members__)}


def defaults_for(project):
    """What each setting is in a new project with this database: DEFAULT_SETTINGS, and for the VAE
    the database's trained values or the recommended latent dimension."""
    q = project.copy()
    q.settings = {}
    q.complete_settings()
    out = {s: dict(v) for s, v in q.settings.items()}
    out["mesh"] = {"n_dof": 30000, "single_layer": False}
    return out


class SettingRow:
    """One setting: its widget, and reading it back."""

    def __init__(self, key, value, choices):
        self.key = key
        self.none = None
        if key in OPTIONAL:
            self.none = QtWidgets.QCheckBox(OPTIONAL[key])
        if key in choices:
            self.widget = QtWidgets.QComboBox()
            self.widget.addItems(choices[key])
        elif key == "saveNet":
            self.widget = QtWidgets.QLineEdit()
        elif isinstance(value, bool) or key in ("single_layer",):
            self.widget = QtWidgets.QCheckBox()
        else:
            self.widget = NumberEdit(0, integer=key in INTEGERS)
        self.set(value)

    def set(self, value):
        if self.none is not None:
            self.none.setChecked(value is None)
            self.widget.setEnabled(value is not None)
            if value is None:
                return
        w = self.widget
        if isinstance(w, QtWidgets.QComboBox):
            w.setCurrentText(str(value))
        elif isinstance(w, QtWidgets.QCheckBox):
            w.setChecked(bool(value))
        elif isinstance(w, NumberEdit):
            w.set_value(value)
        else:
            w.setText(str(value))

    def ok(self):
        if self.none is not None and self.none.isChecked():
            return True
        return not isinstance(self.widget, NumberEdit) or self.widget.ok()

    def get(self):
        if self.none is not None and self.none.isChecked():
            return None
        w = self.widget
        if isinstance(w, QtWidgets.QComboBox):
            return w.currentText()
        if isinstance(w, QtWidgets.QCheckBox):
            return w.isChecked()
        if isinstance(w, NumberEdit):
            return w.value()
        return w.text().strip() or None

    def changed_signal(self, slot):
        w = self.widget
        for signal in ((w.currentTextChanged,) if isinstance(w, QtWidgets.QComboBox) else
                       (w.toggled,) if isinstance(w, QtWidgets.QCheckBox) else (w.textChanged,)):
            signal.connect(lambda *_: slot())
        if self.none is not None:
            self.none.toggled.connect(lambda on: (self.widget.setEnabled(not on), slot()))


class SettingsPanel(QtWidgets.QDialog):
    """Every setting of a run, on two tabs: Optimization (the main ones open, the rest under Advanced,
    which opens on a click) and VAE. A setting not at its default is in bold, with a reset button."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(720, 560)
        self.project = project
        self.defaults = defaults_for(project)
        choices = setting_choices()
        self.rows = {}                                          # (section, key) -> SettingRow, label, reset
        layout = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)
        pages, advanced = {}, None
        for tab, group, keys in LAYOUT:
            if tab not in pages:
                scroll = QtWidgets.QScrollArea()
                scroll.setWidgetResizable(True)
                scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
                inner = QtWidgets.QWidget()
                pages[tab] = QtWidgets.QVBoxLayout(inner)
                scroll.setWidget(inner)
                tabs.addTab(scroll, tab)
            box = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(box)
            grid.setColumnStretch(4, 1)
            grid.setColumnMinimumWidth(0, 270 if group in ("Main", "VAE") else 258)   # every group lines up
            grid.setColumnMinimumWidth(1, 290)
            for r, (section, key) in enumerate(keys):
                row = SettingRow(key, self.current(section, key), choices)
                label = QtWidgets.QLabel(LABELS[key])
                reset = QtWidgets.QToolButton()                 # shown only on a value other than the default
                reset.setText("\u21ba")
                reset.setAutoRaise(True)
                reset.setToolTip(f"Back to the default: {shown_setting(key, self.defaults[section].get(key))}")
                policy = reset.sizePolicy()
                policy.setRetainSizeWhenHidden(True)
                reset.setSizePolicy(policy)
                reset.clicked.connect(lambda _=False, s=section, k=key: self.reset(s, k))
                editor = QtWidgets.QHBoxLayout()
                if row.none is not None:
                    editor.addWidget(row.none)
                editor.addWidget(row.widget)
                if key == "saveNet":                            # always next to the database: no path to type
                    row.widget.setVisible(False)
                elif not isinstance(row.widget, QtWidgets.QCheckBox):
                    row.widget.setFixedWidth(80 if row.none is not None else 140)
                grid.addWidget(label, r, 0)
                grid.addLayout(editor, r, 1, QtCore.Qt.AlignLeft)
                grid.addWidget(reset, r, 2)
                grid.addWidget(Info(HELP.get(key, "")), r, 3)
                self.rows[(section, key)] = (row, label, reset)
                row.changed_signal(lambda s=section, k=key: self._mark(s, k))
            if group == "VAE":                                  # the network, its checks, then the settings
                self.vae_status, self.vae_checks = QtWidgets.QLabel(), QtWidgets.QLabel()
                self.vae_checks.setWordWrap(True)
                pages[tab].addWidget(with_info(self.vae_status, VAE_NOTE))
                pages[tab].addWidget(self.vae_checks)
                pages[tab].addWidget(box)
            elif group == "Main":
                pages[tab].addWidget(box)
            else:                                               # every other group: under Advanced, closed
                if advanced is None:
                    header = QtWidgets.QToolButton()
                    header.setCheckable(True)
                    header.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
                    header.setArrowType(QtCore.Qt.RightArrow)
                    header.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
                    advanced = QtWidgets.QWidget()
                    advanced_layout = QtWidgets.QVBoxLayout(advanced)
                    advanced_layout.setContentsMargins(12, 0, 0, 0)
                    advanced.setVisible(False)
                    header.toggled.connect(lambda on, b=advanced, h=header: (b.setVisible(on), h.setArrowType(
                        QtCore.Qt.DownArrow if on else QtCore.Qt.RightArrow)))
                    pages[tab].addWidget(header)
                    pages[tab].addWidget(advanced)
                    self.rows[("group", "Advanced")] = (header, [])
                title = QtWidgets.QLabel(group)
                title.setStyleSheet(GREY)
                advanced_layout.addWidget(title)
                advanced_layout.addWidget(box)
                self.rows[("group", "Advanced")][1].extend(keys)
        for page in pages.values():
            page.addStretch()
        for (section, key) in list(self.rows):
            if section != "group":
                self._mark(section, key)
        self.vae_synced = []                                    # VAE settings set to the network's
        self.database_ok = database_choice(project)[1] is None and not _database_errors(project)
        self.rows[("vae", "saveNet")][0].widget.setEnabled(False)       # fixed: next to the database
        self.rows[("vae", "saveNet")][0].none.setEnabled(False)
        self.rows[("vae", "use_pretrained_vae")][0].widget.toggled.connect(lambda on: self._vae_sync(on))
        for key in VAE_KEYS:
            self.rows[("vae", key)][0].changed_signal(self._vae_refresh)
        self._vae_sync(None)
        ok_cancel(self, layout)

    # ------------------------------------------------------------------ the VAE tab
    def _vae_values(self):
        out = {}
        for key in ("use_pretrained_vae", "saveNet") + VAE_KEYS:
            row = self.rows[("vae", key)][0]
            out[key] = row.get() if row.ok() else self.current("vae", key)
        return out

    def _vae_project(self):
        q = self.project.copy()
        q.settings["vae"] = self._vae_values()
        return q

    def _vae_sync(self, toggled):
        """With a trained network in use, the VAE settings are its own and greyed out: a shipped
        network's from DATABASE_VAE, another's latent dimension and width read from the file."""
        from lsr_mmto.materials.databases import DATABASE_VAE
        from lsr_mmto.problem.project import network_shape
        use = self.rows[("vae", "use_pretrained_vae")][0]
        if not self.database_ok:
            self._vae_refresh()
            return
        q = self._vae_project()
        nt = q.network_path()
        exists = os.path.isfile(nt) and os.path.getsize(nt) > 0
        if not exists and use.widget.isEnabled():               # no network yet: the first run trains one.
            use.widget.blockSignals(True)                        # Shown unticked; the file keeps its value, so
            use.widget.setChecked(False)                         # the runs after the first use the new network
            use.widget.blockSignals(False)
        use.widget.setEnabled(exists)
        trained = exists and use.get()
        if trained:
            known = dict(DATABASE_VAE.get(q.database, {}))
            shape = network_shape(nt)
            if shape:
                known.update(vae_hiddenDim=shape[1], latentDim=shape[2])
            for key, value in known.items():
                row = self.rows[("vae", key)][0]
                if not same(row.get() if row.ok() else None, value):
                    row.set(value)
                    self.vae_synced.append(key)
                self._mark("vae", key)
        for key in VAE_KEYS:
            row, label, reset = self.rows[("vae", key)]
            row.widget.setEnabled(not trained)
            if trained:
                reset.setEnabled(False)
                reset.setVisible(False)
        self._vae_refresh()

    def _vae_refresh(self):
        """The network line and the live errors and warnings under the note."""
        if not self.database_ok:
            self.vae_status.setText("Network: the database cannot be read, so the network is not checked.")
            self.vae_checks.setText("")
            return
        errors, warnings, status = self._vae_project().vae_check()
        self.vae_status.setText(status)
        lines = [f"<span style='color:#b00000'>Error: {e}</span>" for e in errors] + \
                [f"<span style='color:#a06000'>Warning: {w}</span>" for w in warnings]
        self.vae_checks.setText("<br>".join(lines))

    def current(self, section, key):
        if section == "mesh":
            return getattr(self.project, key)
        return self.project.settings.get(section, {}).get(key, self.defaults[section].get(key))

    def value(self, section, key):
        row = self.rows[(section, key)][0]
        return row.get() if row.ok() else None

    def _mark(self, section, key):
        row, label, reset = self.rows[(section, key)]
        changed = row.ok() and not same(row.get(), self.defaults[section].get(key))
        font = label.font()
        font.setBold(changed)
        label.setFont(font)
        reset.setEnabled(changed or not row.ok())
        reset.setVisible(reset.isEnabled())
        for name, entry in self.rows.items():                   # "Advanced  (2 changed)"
            if name[0] == "group" and (section, key) in entry[1]:
                header, keys = entry
                n = sum(1 for s, k in keys if self.rows[(s, k)][0].ok() and
                        not same(self.rows[(s, k)][0].get(), self.defaults[s].get(k)))
                header.setText(name[1] + (f"  ({n} changed)" if n else ""))

    def reset(self, section, key):
        self.rows[(section, key)][0].set(self.defaults[section].get(key))
        self._mark(section, key)

    def problems(self):
        return [f"{LABELS[k]} is not a number" for (s, k), entry in self.rows.items()
                if s != "group" and not entry[0].ok()]

    def accept(self):
        problems = self.problems()
        if problems:
            QtWidgets.QMessageBox.warning(self, "Settings", "\n".join(problems))
            return
        super().accept()

    def apply(self, project):
        """Every setting as the panel shows it, each unchanged one exactly as it was."""
        changed = []
        for (section, key), entry in self.rows.items():
            if section == "group" or not entry[0].widget.isEnabled() and key == "use_pretrained_vae":
                continue                                         # greyed for want of a network: kept as it is
            new, old = entry[0].get(), self.current(section, key)
            if same(new, old) and (new is None) == (old is None):
                continue
            changed.append(f"{LABELS[key]} {shown_setting(key, old)} -> {shown_setting(key, new)}")
            if section == "mesh":
                setattr(project, key, int(new) if key == "n_dof" else bool(new))
            else:
                project.settings.setdefault(section, {})[key] = new
        out = ["Settings: " + ", ".join(changed)] if changed else []
        if self.vae_synced:
            out.append(f"{', '.join(LABELS[k] for k in dict.fromkeys(self.vae_synced))}: set to the trained "
                       f"network's")
        return out


def all_setting_keys():
    """(section, key) of every setting the panel shows."""
    return [sk for _, _, keys in LAYOUT for sk in keys]
