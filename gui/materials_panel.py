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
                "kept.\n\nA limit written for the old numbers means something else afterwards; converting it keeps "
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


INCLUDED = " (included)"                     # after the name of a database that comes with LSR MMTO
NO_DATABASE = "No database yet. Make one with New database, or bring one in with Import."


def import_help():
    """The Import (i): what a table brought in must look like and what happens to it; the properties it
    reads are listed from the registry (databases.ATTRIBUTES), each with its shortest other name."""
    from lsr_mmto.materials.databases import ATTRIBUTES
    names = ", ".join(a.name + (f" (or {min(a.aliases, key=len)})" if a.aliases else "") for a in ATTRIBUTES.values())
    return ("Import brings in a table of materials made elsewhere, as an Excel or CSV file.\n\n"
            "**The file:** material names in the first column, property names in the first row, each with its "
            "unit in brackets, e.g. Density [kg/m^3]. The units may go in a second row instead.\n\n"
            "**What happens:**\n\n"
            "1. A database made by LSR MMTO is used as it is.\n"
            f"2. Any other table: only columns named after a property LSR MMTO knows are read: {names}. A message "
            "lists the columns left out. To keep one, rename it in your file and import again.\n"
            "3. The editor opens. Check every value, choose any unit it could not read, then Save. The database "
            "is kept in the project's folder.\n"
            "4. If its units differ from the project's, use Convert units below the table.\n"
            "5. It has no trained network yet, so the first Run trains one (a few minutes). Then look at Network "
            "error.\n\n"
            "Every property, its other names and its units: Help > Researcher guide > Reference.")


def database_choices(project):
    """[(value, text, where)] for the Database list: every database in the project's materials folder, then
    the project's own database when it lives elsewhere. value is what project.database holds; text what the
    list shows (a database that comes with LSR MMTO as '<name> (included)'); where its file."""
    from lsr_mmto.materials.databases import DATABASE_VAE
    from lsr_mmto.problem.project_folder import MATERIALS
    folder = os.path.join(project.base_dir, MATERIALS)
    out = []
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder), key=str.lower):
            if name.lower().endswith(".xlsx") and not name.startswith("~$"):
                out.append((f"{MATERIALS}/{name}", os.path.splitext(name)[0], os.path.join(folder, name)))
    if project.database is not None:
        here = os.path.normcase(os.path.abspath(project.database_paths()[0]))
        if not any(os.path.normcase(os.path.abspath(where)) == here for _, _, where in out):
            out.append(choice_of(project, project.database))
    return out


def choice_of(project, value):
    """(value, text, where) of one database, as database_choices lists it."""
    from lsr_mmto.materials.databases import DATABASE_VAE
    where = project.copy(_database=value).database_paths()[0]
    text = value + INCLUDED if value in DATABASE_VAE else os.path.splitext(os.path.basename(where))[0]
    return value, text, where


class FittedCombo(QtWidgets.QComboBox):
    """A drop-down whose open list is exactly as wide as itself, long names cut short with ..."""

    def showPopup(self):
        self.view().setTextElideMode(QtCore.Qt.ElideRight)
        super().showPopup()
        popup = self.view().window()
        popup.setFixedWidth(self.width())
        popup.move(self.mapToGlobal(QtCore.QPoint(0, 0)).x(), popup.y())


class MaterialsPanel(QtWidgets.QDialog):
    """The material database: which one, its materials in their colours and labels, its check,
    and the latent dimension recommended for it. The Database list holds this project's own databases;
    those that come with LSR MMTO are under Included databases. Choosing another database resets the VAE
    settings to the new one's (its trained values, or the recommended latent dimension); a trained
    network sets the shape settings from itself (Project.match_network). When the window becomes active
    again and the database's file changed (saved in Excel, say), it is read again."""

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
        self.combo = FittedCombo()
        for value, text, where in database_choices(project):
            self._add_choice(value, text, where)
        self.combo.setCurrentIndex(-1 if project.database is None else self.combo.findData(project.database))
        self.combo.currentIndexChanged.connect(lambda _: self._choose(self.combo.currentData()))
        row.addWidget(self.combo, stretch=1)
        self.included = QtWidgets.QPushButton("Included databases")
        self.included.setProperty("hasMenu", True)
        menu = QtWidgets.QMenu(self.included)
        for name in DATABASE_VAE:
            menu.addAction(name, lambda n=name: self.choose(n))
        self.included.setMenu(menu)
        row.addWidget(self.included)
        layout.addLayout(row)

        actions = QtWidgets.QHBoxLayout()
        self.new_button = QtWidgets.QPushButton("New database...")
        self.edit_button = QtWidgets.QPushButton("Edit...")
        self.excel_button = QtWidgets.QPushButton("Open in Excel")
        self.import_button = QtWidgets.QPushButton("Import...")
        for button, slot in ((self.new_button, self.new_database), (self.edit_button, self.edit_database),
                             (self.excel_button, self._open_in_excel), (self.import_button, self.import_database)):
            button.clicked.connect(lambda _=False, s=slot: s())     # not s(checked): Qt's checked is no path
            if button is self.import_button:
                actions.addStretch()
            actions.addWidget(button)
        actions.addWidget(Info(import_help()))
        layout.addLayout(actions)

        shown = QtWidgets.QVBoxLayout()                             # the tabs right on the table
        shown.setSpacing(0)
        switch = QtWidgets.QHBoxLayout()
        self.tabs = QtWidgets.QTabBar()
        self.tabs.setObjectName("underlineTabs")
        self.tabs.setDrawBase(False)
        self.tabs.setExpanding(False)
        for text in ("Values", "Network error"):
            self.tabs.addTab(text)
        self.tabs.currentChanged.connect(lambda _: self._fill_table())
        switch.addWidget(self.tabs)
        switch.addWidget(Info("**Values:** each material's properties as the database gives them.\n\n"
                              "**Network error:** how far the trained network is from each value when it turns the "
                              "material into its point on the material map and back, in % of the property's "
                              "range over the database's materials.\n\nRest the mouse on a cell for both values "
                              "and the difference in the property's unit."))
        switch.addStretch()
        shown.addLayout(switch)
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        shown.addWidget(self.table, stretch=1)
        layout.addLayout(shown, stretch=1)
        self._shown_table = None                                    # what _refresh read, for _fill_table
        self._read = None                                           # the files' times when they were read
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
        self.new_button.setDefault(True)
        self._refresh()

    def _add_choice(self, value, text, where):
        self.combo.addItem(text, value)
        self.combo.setItemData(self.combo.count() - 1, where, QtCore.Qt.ToolTipRole)

    def choose(self, value):
        """The database value (as project.database holds it) chosen, added to the list when it is not there."""
        if self.combo.findData(value) < 0:
            self._add_choice(*choice_of(self.project, value))
        self.combo.setCurrentIndex(self.combo.findData(value))

    def _choose(self, value):
        if value and value != self.project.database:
            self.project.database = value
            self._refresh()

    def showEvent(self, event):
        super().showEvent(event)
        self._read = self._file_times()

    def event(self, event):
        """Active again (back from Excel, say) with the database's file changed: read it again."""
        if event.type() == QtCore.QEvent.WindowActivate and self._read is not None and \
                self._file_times() != self._read:
            self._refresh()
        return super().event(event)

    def _file_times(self):
        """When the database's file and its network were last changed (None for one that is not there)."""
        if self.project.database is None:
            return None
        xlsx = self.project.database_paths()[0]
        return tuple(os.path.getmtime(f) if os.path.isfile(f) else None for f in (xlsx, self.project.network_path()))

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
        """A table of materials made elsewhere (an Excel or CSV file). One that is already a database of LSR
        MMTO (its check finds nothing wrong) is used where it is, with its trained network beside it if it has
        one. Any other: the columns that are properties LSR MMTO knows are read, the others listed in one
        message; the database editor then opens with what was read, to be checked and saved in the project's
        folder, which the project then uses. Returns the database's path, or None."""
        from lsr_mmto.materials.databases import check_database, import_table
        from .database_editor import DatabaseEditor
        from .pyto_ad_imports import FILE_DIALOG_OPTIONS
        if path is None:
            window = self.parent()
            start = window.database_folder() if hasattr(window, "own_folder") and window.own_folder() else \
                self.project.base_dir
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import a table of materials", start,
                                                            "Tables (*.xlsx *.xls *.csv)", options=FILE_DIALOG_OPTIONS)
            if not path:
                return None
        if path.lower().endswith(".xlsx") and not check_database(path)[0]:
            self.set_database(path)                                 # a database of LSR MMTO: used where it is
            return path
        folder = self._own_folder("Import a database")
        if folder is None:
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
        self.choose(project.database)
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
        self.choose(name)

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
        self.tabs.setTabEnabled(1, False)
        self._read = self._file_times()
        self.convert_button.setText(f"Convert to {self.project.units} units...")
        self.convert_button.setVisible(self.project.database == self.original and
                                       bool(self.project.unit_conversion()[0]))
        for button in (self.edit_button, self.excel_button):
            button.setEnabled(self.project.database is not None)
        if self.project.database is None:
            self.report.setText(NO_DATABASE)
            return
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
        if self._network_rows is None and self.tabs.currentIndex() == 1:
            self.tabs.setCurrentIndex(0)                            # fills the table
        else:
            self._fill_table()
        self.report.setText("\n".join(lines))

    def _fill_table(self):
        """The table of the materials: their values, or with Network error chosen, how far the trained network
        is from each value (in % of the property's range; both values and the difference behind the cell)."""
        from lsr_mmto.materials.databases import SHOWN, attribute_name
        if self._shown_table is None:
            return
        names, attributes, units, values, colours, labels = self._shown_table
        errors = self.tabs.currentIndex() == 1 and self._network_rows is not None
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
            self.tabs.setTabEnabled(1, True)
        return line

    def apply(self, project):
        """The database, and on a change the VAE settings of the new one. Returns lines saying what changed."""
        if self.project.database == self.original:
            return []
        q = self.project.copy()
        q.settings["vae"] = {"use_pretrained_vae": project.settings["vae"].get("use_pretrained_vae", True)}
        q.complete_settings()
        matched = q.match_network()                                 # a trained network gives its own shape
        before = {k: project.settings["vae"].get(k) for k in VAE_KEYS}
        cleared = project.clear_hand_set_network()                  # it belonged to the database before
        project.settings["vae"] = q.settings["vae"]
        project.database = q.database
        changed = [f"{BY_KEY[k].label} {before[k]} -> {q.settings['vae'][k]}" for k in before
                   if not same(before[k], q.settings["vae"][k])]
        return [f"Database: {q.database}"] + ([f"VAE settings for it: {', '.join(changed)}"] if changed else []) + \
            [line for line in (cleared, matched) if line]
