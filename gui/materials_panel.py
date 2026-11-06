"""
materials_panel.py
==================
The Materials panel of the LSR MMTO window: the material database and its network.
Like every panel, a dialog with OK and Cancel working on a copy of the project: OK hands back
the edited parts (apply), Cancel leaves the project as it was. A value left as it was is handed
back exactly as it was, so a panel closed with OK and no edits changes nothing in the file.
"""
import json
import math
import os
import subprocess
import sys

from PyQt5 import QtCore, QtWidgets

from lsr_mmto.problem.settings import BY_KEY, VAE_PARAMS as VAE_KEYS

from .panel_widgets import Info, database_choice, ok_cancel, same, swatch


CONVERT_HELP = ("The database is written again as the project's own copy, in its materials folder, with these "
                "numbers in the project's units. Its trained network still fits the converted numbers, so it is "
                "kept. A limit written for the old numbers means something else afterwards; converting it keeps "
                "the limit the same in the converted units.")


def conversion_lines(project, columns, limits):
    """(column lines, limit lines) of a unit conversion (Project.unit_conversion), in words: "Young's
    modulus: Pa -> MPa, x 1e-06", "Mass <= 4e+07 -> 4e-05 t"."""
    from lsr_mmto.materials.databases import SHOWN
    constraints = project.formulation.get("constraints", [])
    return ([f"{SHOWN[name]}: {unit} -> {to}, x {factor:g}" for name, unit, to, factor in columns],
            [f"{q.shown} {constraints[i]['op']} {bound:g} -> {new:g}{project.quantity_unit(q)}"
             for i, q, bound, new in limits])


class ConvertUnitsDialog(QtWidgets.QDialog):
    """Convert to <units> units: each column of the database in other units, with its factor, and the limits
    written for the old numbers with what they become, converted too while "Convert these limits too" is
    ticked."""

    def __init__(self, parent, project, columns, limits):
        super().__init__(parent)
        self.setWindowTitle("Convert units")
        layout = QtWidgets.QVBoxLayout(self)
        column_lines, limit_lines = conversion_lines(project, columns, limits)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(QtWidgets.QLabel(f"The database in {project.units} units:"))
        head.addWidget(Info(CONVERT_HELP))
        head.addStretch()
        layout.addLayout(head)
        for line in column_lines:
            layout.addWidget(QtWidgets.QLabel("    " + line))
        self.limits = QtWidgets.QCheckBox("Convert these limits too")
        self.limits.setChecked(True)
        if limit_lines:
            layout.addWidget(QtWidgets.QLabel("Limits written for the old numbers:"))
            for line in limit_lines:
                layout.addWidget(QtWidgets.QLabel("    " + line))
            layout.addWidget(self.limits)
        else:
            self.limits.setChecked(False)
        ok_cancel(self, layout)


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
        for text, slot in (("New database...", self.new_database), ("Import...", self.import_database),
                           ("Edit...", self.edit_database),
                           ("Browse...", self._browse), ("Open in Excel", self._open_in_excel),
                           ("Check again", self._refresh)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
        layout.addLayout(row)

        switch = QtWidgets.QHBoxLayout()
        self.show_values = QtWidgets.QPushButton("Values")
        self.show_errors = QtWidgets.QPushButton("Network error")
        self.shown = QtWidgets.QButtonGroup(self)
        for b in (self.show_values, self.show_errors):
            b.setCheckable(True)
            self.shown.addButton(b)
            switch.addWidget(b)
        self.show_values.setChecked(True)
        self.shown.buttonClicked.connect(lambda _: self._fill_table())
        switch.addWidget(Info("Values: each material's properties as the database gives them.\n"
                              "Network error: how far the trained network is from each value when it turns the "
                              "material into its point on the material map and back, in % of the property's "
                              "range over the database's materials. Rest the mouse on a cell for both values "
                              "and the difference in the property's unit."))
        switch.addStretch()
        layout.addLayout(switch)
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table, stretch=1)
        self._shown_table = None                                    # what _refresh read, for _fill_table
        self.report = QtWidgets.QTextEdit()
        self.report.setReadOnly(True)
        self.report.setFixedHeight(64)
        below = QtWidgets.QHBoxLayout()
        below.addWidget(self.report, stretch=1)
        self.latent_info = Info("")                                 # how the latent dimension is recommended
        self.latent_info.setVisible(False)
        below.addWidget(self.latent_info, alignment=QtCore.Qt.AlignTop)
        layout.addLayout(below)
        self.convert_button = QtWidgets.QPushButton("")               # only while the units differ
        self.convert_button.clicked.connect(lambda: self.convert_units())
        layout.addWidget(self.convert_button, alignment=QtCore.Qt.AlignLeft)
        ok_cancel(self, layout)
        self._refresh()

    def _choose(self, name):
        if name:
            self.project.database = name
            self._refresh()

    def _browse(self):
        from lsr_mmto.materials.databases import DATABASE_DIR
        from .pyto_ad_imports import FILE_DIALOG_OPTIONS
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

    def import_database(self, path=None):
        """A database made from a table of materials made elsewhere (an Excel or CSV file): the columns that
        are properties LSR MMTO knows are read, the others listed in one message; the database editor then
        opens with what was read, to be checked and saved in the project's folder, which the project then
        uses."""
        from lsr_mmto.materials.databases import import_table
        from .database_editor import DatabaseEditor
        from .pyto_ad_imports import FILE_DIALOG_OPTIONS
        folder = self._own_folder("Import a database")
        if folder is None:
            return None
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import a table of materials", folder,
                                                            "Tables (*.xlsx *.xls *.csv)", options=FILE_DIALOG_OPTIONS)
            if not path:
                return None
        try:
            table = import_table(path)
        except Exception as e:                                      # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Import a database", f"Nothing was imported: {e}.")
            return None
        table["stem"] = os.path.splitext(os.path.basename(path))[0]
        note = self.import_note(table)
        if note:
            QtWidgets.QMessageBox.information(self, "Import a database", note)
        editor = DatabaseEditor(self, folder, unit_system=self.project.units, imported=table)
        if not editor.exec_():
            return None
        saved = editor.saved_path or editor.save()
        self.set_database(saved)
        self.report.setText(f"{os.path.basename(saved)} saved in the project's folder.\n" + self.report.toPlainText())
        return saved

    @staticmethod
    def import_note(table):
        """What an import read and what it left out, in plain words; None when it took every column."""
        from lsr_mmto.materials.databases import SHOWN
        if not table["skipped"] and not table["unknown_units"]:
            return None
        lines = [f"Read {len(table['names'])} materials with {', '.join(SHOWN[a] for a in table['attributes'])}."]
        if table["skipped"]:
            lines.append("Not imported, as LSR MMTO has no property of that name: "
                         + ", ".join(table["skipped"]) + ".")
        for key, unit in table["unknown_units"]:
            lines.append(f"{SHOWN[key]}: the unit {unit} is not one LSR MMTO knows; choose its unit in the "
                         f"editor's units row.")
        lines.append("Check the values in the editor, then Save.")
        return "\n".join(lines)

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

    def convert_units(self):
        """Convert to <units> units...: the project's database (the one it uses now) converted into the
        project's units, and the limits too when ticked, at once (the window's convert_database_units). The
        panel then shows the converted database. Returns what was said in the messages, or None."""
        window = self.parent()
        if self._own_folder("Convert units") is None:
            return None
        lines = window.convert_database_units(self)
        if not lines:
            return None
        project = window.project
        self.original = self.project.database = project.database
        self.project.formulation = json.loads(json.dumps(project.formulation))
        self.project.settings["vae"] = dict(project.settings["vae"])
        self.combo.blockSignals(True)
        if self.combo.findText(project.database) < 0:
            self.combo.addItem(project.database)
        self.combo.setCurrentText(project.database)
        self.combo.blockSignals(False)
        self._refresh()
        return lines

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
        self._shown_table, self._network_rows = None, None
        self.latent_info.setVisible(False)
        self.show_errors.setEnabled(False)
        self.convert_button.setText(f"Convert to {self.project.units} units...")
        self.convert_button.setVisible(self.project.database == self.original and
                                       bool(self.project.unit_conversion()[0]))
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
        self._shown_table = (names, attributes, units, values, colours, labels)
        if not errors:
            d_eff, n_z, ratios = latent_dim_breakdown(xlsx)
            lines.insert(0, f"{len(names)} materials; latent dimension {n_z} recommended (effective dimension "
                            f"{effective_text(d_eff, n_z)}).")
            self.latent_info.setToolTip(latent_help(d_eff, n_z, ratios))
            self.latent_info.setVisible(True)
            network = self._network_line()
            if network:
                lines.insert(1, network)
        if self._network_rows is None and self.show_errors.isChecked():
            self.show_values.setChecked(True)
        self._fill_table()
        self.report.setText("\n".join(lines))

    def _fill_table(self):
        """The table of the materials: their values, or with Network error chosen, how far the trained network
        is from each value (in % of the property's range; both values and the difference behind the cell)."""
        from lsr_mmto.materials.databases import SHOWN, attribute_name
        if self._shown_table is None:
            return
        names, attributes, units, values, colours, labels = self._shown_table
        errors = self.show_errors.isChecked() and self._network_rows is not None
        by_cell = {(r["material"], attribute_name(r["property"])): r for r in (self._network_rows or [])}
        heads = ["Label", "Material"] + [SHOWN.get(a, a) + ("" if errors or u is None else f" [{u}]")
                                         for a, u in zip(attributes, units)]
        self.table.clear()
        self.table.setColumnCount(len(heads))
        self.table.setHorizontalHeaderLabels(heads)
        self.table.setRowCount(len(names))
        for i, name in enumerate(names):
            item = QtWidgets.QTableWidgetItem(labels[i])
            if colours[i]:
                item.setIcon(swatch(colours[i]))
            self.table.setItem(i, 0, item)
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(name))
            for j, a in enumerate(attributes):
                r = by_cell.get((name, a))
                if errors and r is not None:
                    pct = r["percent_of_range"]
                    cell = QtWidgets.QTableWidgetItem("< 0.01%" if pct < 0.01 else f"{pct:.2g}%")
                    unit = f" {units[j]}" if units[j] else ""
                    cell.setToolTip(f"Database: {r['database']:.6g}{unit}\nNetwork: {r['network']:.6g}{unit}\n"
                                    f"Difference: {r['difference']:+.3g}{unit}")
                    cell.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
                else:
                    cell = QtWidgets.QTableWidgetItem("" if errors else format(values[i, j], ".6g"))
                self.table.setItem(i, 2 + j, cell)
        self.table.verticalHeader().setVisible(False)
        self.table.setColumnHidden(1, list(labels) == list(names))      # Material: only when the labels differ
        self.table.resizeColumnsToContents()

    def _network_line(self):
        """How closely the trained network gives back the database's materials, in one line (every value
        under Network error); None when the database has no trained network yet."""
        rows, line = self.project.network_error()
        if rows is not None:
            self._network_rows = rows
            self.show_errors.setEnabled(True)
        return line

    def apply(self, project):
        """The database, and on a change the VAE settings of the new one. Returns lines saying what changed."""
        if self.project.database == self.original:
            return []
        q = self.project.copy()
        q.settings["vae"] = {"use_pretrained_vae": project.settings["vae"].get("use_pretrained_vae", True)}
        q.complete_settings()
        before = {k: project.settings["vae"].get(k) for k in VAE_KEYS + ("saveNet",)}
        project.database = q.database
        project.settings["vae"] = q.settings["vae"]
        changed = [f"{BY_KEY[k].label} {before[k]} -> {q.settings['vae'][k]}" for k in before
                   if not same(before[k], q.settings["vae"][k])]
        return [f"Database: {q.database}"] + ([f"VAE settings for it: {', '.join(changed)}"] if changed else [])
