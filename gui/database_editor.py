"""
database_editor.py
==================
A material database made or changed in the app: one row per material (its name, an optional
shorter name to show, its colour, then a number for each property) and a units row under the
headers. It saves the same Excel file as before (databases.write_database), so Excel can still
open it. A database that comes with LSR MMTO is never changed: it is saved as a copy in the
project's folder. When the numbers change, the network trained on the old numbers no longer
fits: it is set aside as <name>.previous.nt and the next run trains a new one.
"""
import os

from PyQt5 import QtCore, QtGui, QtWidgets

from .panels import GREY, Info, ok_cancel

# how the properties are shown (any other in the registry is shown by its own name), and those a new
# database starts with; Add property... lists every property of the registry (databases.ATTRIBUTES)
SHOWN = {"Density": "Density", "Youngs_Modulus": "Young's modulus", "Poissons_Ratio": "Poisson's ratio",
         "Yield_Strength": "Yield strength", "Cost": "Cost"}
STARTING = ["Density", "Youngs_Modulus", "Poissons_Ratio"]
FIXED = ["Name", "Shown as", "Colour"]                      # the columns before the properties
BAD = QtGui.QColor("#ffd6d6")
HELP = ("One row per material. Shown as is optional: a shorter name for legends and plots (the name is "
        "shown when it is empty). Click a colour to change it. The row under the headers holds each "
        "property's unit. Add property adds a column; Remove property removes the column you are in. "
        "You can paste a block copied from Excel into the table.")


class _Table(QtWidgets.QTableWidget):
    """A table that takes a block pasted from Excel (tab between cells, a new line between rows), adding
    material rows as needed."""
    pasted = QtCore.pyqtSignal()

    def keyPressEvent(self, event):
        if event.matches(QtGui.QKeySequence.Paste):
            self.paste(QtWidgets.QApplication.clipboard().text())
            return
        if event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace) and self.state() != self.EditingState:
            for item in self.selectedItems():
                if item.row() > 0 and item.column() != 2:
                    item.setText("")
            return
        super().keyPressEvent(event)

    def paste(self, text):
        rows = [line.split("\t") for line in text.replace("\r\n", "\n").rstrip("\n").split("\n")]
        r0, c0 = max(self.currentRow(), 1), max(self.currentColumn(), 0)
        for i, values in enumerate(rows):
            r = r0 + i
            while r >= self.rowCount():
                self.parent_editor.add_material()
            for j, value in enumerate(values):
                c = c0 + j
                if c < self.columnCount() and c != 2:
                    self.item(r, c).setText(value.strip())
        self.pasted.emit()


class DatabaseEditor(QtWidgets.QDialog):
    """New (path None: a database to be saved in folder) or an existing database to change. A shipped one
    (in_shipped_folder) opens as a copy to be saved in folder. saved_path is set by a successful Save, and
    network_set_aside names the network moved out of the way when the numbers changed."""

    def __init__(self, parent, folder, path=None, unit_system=None):
        from databases import in_shipped_folder  # type: ignore
        super().__init__(parent)
        path = None if path is None else os.path.normpath(os.path.abspath(path))   # one kind of slash on Windows
        self.folder, self.path, self.unit_system = folder, path, unit_system
        self.copy = path is not None and in_shipped_folder(path)
        self.saved_path, self.network_set_aside, self.loaded = None, None, None
        self.setWindowTitle("New database" if path is None else "Database")
        self.resize(900, 520)
        layout = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Name"))
        self.name = QtWidgets.QLineEdit()
        top.addWidget(self.name, stretch=1)
        self.where = QtWidgets.QLabel()
        self.where.setStyleSheet(GREY)
        top.addWidget(self.where, stretch=2)
        top.addWidget(Info(HELP))
        layout.addLayout(top)

        self.table = _Table()
        self.table.parent_editor = self
        self.table.verticalHeader().setVisible(False)
        self.table.cellClicked.connect(self._clicked)
        self.table.itemChanged.connect(lambda _: self.check())
        self.table.pasted.connect(self.check)
        layout.addWidget(self.table, stretch=1)

        rows = QtWidgets.QHBoxLayout()
        for text, slot in (("Add material", self.add_material), ("Remove material", self.remove_material)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            rows.addWidget(button)
        rows.addSpacing(24)
        self.add_property_button = QtWidgets.QPushButton("Add property...")
        self.add_property_menu = QtWidgets.QMenu(self)
        self.add_property_menu.aboutToShow.connect(self._fill_property_menu)
        self.add_property_button.setMenu(self.add_property_menu)
        rows.addWidget(self.add_property_button)
        remove = QtWidgets.QPushButton("Remove property")
        remove.clicked.connect(lambda: self.remove_property())
        rows.addWidget(remove)
        rows.addStretch()
        layout.addLayout(rows)
        self.problems_text = QtWidgets.QLabel()
        self.problems_text.setWordWrap(True)
        self.problems_text.setStyleSheet("color: #b00000;")
        layout.addWidget(self.problems_text)
        self.buttons = ok_cancel(self, layout)
        self.save_button = self.buttons.button(QtWidgets.QDialogButtonBox.Ok)
        self.save_button.setText("Save")

        self.attributes = []
        if path is None:
            self.name.setText(self._free_name("my_materials"))
            self._columns(order=STARTING)
            for _ in range(3):
                self.add_material()
        else:
            self._load(path)
        self.name.setReadOnly(path is not None and not self.copy)
        self._where()
        self.name.textChanged.connect(lambda _: (self._where(), self.check()))
        self.check()

    # ------------------------------------------------------------------ reading
    def _free_name(self, stem):
        name, k = stem, 2
        while os.path.exists(os.path.join(self.folder, name + ".xlsx")):
            name, k = f"{stem} {k}", k + 1
        return name

    def _load(self, path):
        from databases import database_table, material_display  # type: ignore
        names, attributes, units, values = database_table(path)
        colours, labels = material_display(path)
        self.loaded = (list(attributes), list(units), values.tolist())
        stem = os.path.splitext(os.path.basename(path))[0]
        self.name.setText(self._free_name(f"{stem} copy") if self.copy else stem)
        self.table.blockSignals(True)
        self._columns(order=attributes, units=units)
        for name, label, colour, row in zip(names, labels, colours, values.tolist()):
            r = self._new_row(colour)
            self.table.item(r, 0).setText(name)
            self.table.item(r, 1).setText("" if label == name else label)
            for j, v in enumerate(row):
                self.table.item(r, 3 + j).setText(format(v, ".12g"))
        self.table.blockSignals(False)

    # ------------------------------------------------------------------ the table
    def _default_unit(self, key):
        from databases import ATTRIBUTES, UNIT_SYSTEMS  # type: ignore
        a = ATTRIBUTES[key]
        return UNIT_SYSTEMS.get(self.unit_system or "", {}).get(a.measures, a.units[0])

    def _fill_property_menu(self):
        from databases import ATTRIBUTES  # type: ignore
        self.add_property_menu.clear()
        missing = [key for key in ATTRIBUTES if key not in self.attributes]
        for key in missing:
            self.add_property_menu.addAction(SHOWN.get(key, key), lambda k=key: self.add_property(k))
        if not missing:
            self.add_property_menu.addAction("Every property is in the table").setEnabled(False)

    def add_property(self, key):
        """A column for the property, at the end, empty."""
        if key not in self.attributes:
            self._columns(order=self.attributes + [key])

    def remove_property(self, key=None):
        """The property's column (by default the one the current cell is in) gone, after a yes when it holds
        numbers. True when it was removed."""
        if key is None:
            c = self.table.currentColumn()
            if c < len(FIXED):
                QtWidgets.QMessageBox.information(self, "Remove property", "Click a cell in the property's "
                                                                           "column first.")
                return False
            key = self.attributes[c - len(FIXED)]
        j = self.attributes.index(key)
        filled = any(self.table.item(r, 3 + j) and self.table.item(r, 3 + j).text().strip()
                     for r in range(1, self.table.rowCount()))
        if filled and QtWidgets.QMessageBox.question(
                self, "Remove property", f"Remove {SHOWN.get(key, key)} and its numbers?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            return False
        self._columns(order=[k for k in self.attributes if k != key])
        return True

    def _columns(self, *_, order=None, units=None):
        """The property columns, in order (by default as they are); a column that stays keeps its numbers
        and unit."""
        from databases import ATTRIBUTES  # type: ignore
        keep = {}
        for j, key in enumerate(self.attributes):
            combo = self.table.cellWidget(0, 3 + j)
            keep[key] = (combo.currentText() if combo else None,
                         [self.table.item(r, 3 + j).text() for r in range(1, self.table.rowCount())])
        wanted = list(order) if order is not None else list(self.attributes)
        was = self.table.blockSignals(True)
        self.table.setColumnCount(len(FIXED) + len(wanted))
        self.table.setHorizontalHeaderLabels(FIXED + [SHOWN.get(k, k) for k in wanted])
        if self.table.rowCount() == 0:
            self.table.setRowCount(1)
        for c in range(len(FIXED)):
            item = QtWidgets.QTableWidgetItem("Units" if c == 0 else "")
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            item.setForeground(QtGui.QColor("#777777"))
            self.table.setItem(0, c, item)
        for j, key in enumerate(wanted):
            combo = QtWidgets.QComboBox()
            combo.addItems(list(ATTRIBUTES[key].units))
            unit = (units[j] if units else None) or keep.get(key, (None,))[0] or self._default_unit(key)
            if combo.findText(unit) < 0:
                combo.addItem(unit)
            combo.setCurrentText(unit)
            combo.currentTextChanged.connect(lambda _: self.check())
            self.table.setCellWidget(0, 3 + j, combo)
            old = keep.get(key, (None, []))[1]
            for r in range(1, self.table.rowCount()):
                self.table.setItem(r, 3 + j, QtWidgets.QTableWidgetItem(old[r - 1] if r - 1 < len(old) else ""))
        self.attributes = list(wanted)
        self.table.blockSignals(was)
        for c, width in enumerate([180, 90, 60] + [145] * len(wanted)):
            self.table.setColumnWidth(c, width)
        self.check()

    def _new_row(self, colour=None):
        from databases import default_colours  # type: ignore
        r = self.table.rowCount()
        self.table.insertRow(r)
        colour = colour or default_colours(r)[r - 1]
        for c in range(self.table.columnCount()):
            self.table.setItem(r, c, QtWidgets.QTableWidgetItem(""))
        self._set_colour(r, colour)
        return r

    def add_material(self):
        was = self.table.blockSignals(True)
        r = self._new_row()
        self.table.blockSignals(was)
        self.check()
        return r

    def remove_material(self):
        r = self.table.currentRow()
        if r >= 1 and self.table.rowCount() > 2:
            self.table.removeRow(r)
            self.check()

    def _set_colour(self, r, colour):
        item = self.table.item(r, 2)
        item.setData(QtCore.Qt.UserRole, colour)
        item.setBackground(QtGui.QColor(colour))
        item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        item.setToolTip(f"{colour}: click to choose another colour")

    def _clicked(self, r, c):
        if r >= 1 and c == 2:
            colour = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.colour(r)), self,
                                                     "Colour of " + (self.table.item(r, 0).text() or "the material"))
            if colour.isValid():
                self._set_colour(r, colour.name())

    def colour(self, r):
        return self.table.item(r, 2).data(QtCore.Qt.UserRole)

    # ------------------------------------------------------------------ checking and saving
    def contents(self):
        """(names, labels, colours, units, rows of text) as typed."""
        rows = range(1, self.table.rowCount())
        cell = lambda r, c: self.table.item(r, c).text().strip() if self.table.item(r, c) else ""  # noqa: E731
        units = [self.table.cellWidget(0, 3 + j).currentText() for j in range(len(self.attributes))]
        return ([cell(r, 0) for r in rows], [cell(r, 1) for r in rows], [self.colour(r) for r in rows], units,
                [[cell(r, 3 + j) for j in range(len(self.attributes))] for r in rows])

    def target(self):
        if self.path is not None and not self.copy:
            return self.path
        return os.path.join(self.folder, self.name.text().strip() + ".xlsx")

    def _where(self):
        self.where.setText(f"Saved in {self.folder}" if self.path is None or self.copy else
                           f"{self.path}")

    def check(self):
        """Every cell checked; a bad one is shaded with the reason on it. Save only with none. The problems."""
        from databases import ATTRIBUTES  # type: ignore
        if self.table.columnCount() < len(FIXED):
            return ["no columns"]
        names, labels, colours, units, rows = self.contents()
        out = []
        was = self.table.blockSignals(True)
        for r, name in enumerate(names, start=1):
            bad = not name or names.count(name) > 1
            self._mark(r, 0, bad, "A name, different from the others" if bad else "")
            for j, key in enumerate(self.attributes):
                a, text = ATTRIBUTES[key], rows[r - 1][j]
                try:
                    value = float(text)
                    reason = "" if a.in_range(value) else f"{SHOWN.get(key, key)}: {a.range_text()}"
                except ValueError:
                    reason = f"{SHOWN.get(key, key)}: a number" if text else f"{SHOWN.get(key, key)}: empty"
                self._mark(r, 3 + j, bool(reason), reason)
                if reason:
                    out.append(f"{name or 'row ' + str(r)}: {reason}")
        self.table.blockSignals(was)
        if any(not n or names.count(n) > 1 for n in names):
            out.insert(0, "Every material needs a name of its own")
        if len(names) < 2:
            out.append("At least two materials")
        if not self.attributes:
            out.append("Add at least one property")
        file_name = self.name.text().strip()
        if (self.path is None or self.copy) and (not file_name or any(ch in file_name for ch in '\\/:*?"<>|')):
            out.append("Name: letters, digits, spaces, - and _")
        elif (self.path is None or self.copy) and os.path.exists(self.target()):
            out.append(f"{os.path.basename(self.target())} is already there; choose another name")
        self.problems_text.setText(out[0] + (f" (and {len(out) - 1} more)" if len(out) > 1 else "") if out else "")
        self.save_button.setEnabled(not out)
        return out

    def _mark(self, r, c, bad, reason):
        item = self.table.item(r, c)
        if item is None:
            return
        item.setBackground(BAD if bad else QtGui.QBrush())
        item.setToolTip(reason)

    def numbers_changed(self, units, rows):
        """True when what a network learns from (the properties, their units, the numbers in order) changed."""
        if self.loaded is None:
            return False
        attributes, old_units, old_values = self.loaded
        new_values = [[float(v) for v in row] for row in rows]
        return attributes != self.attributes or old_units != units or old_values != new_values

    def save(self):
        """Write the database; when its numbers changed, set its trained network aside. The path written."""
        from databases import write_database  # type: ignore
        names, labels, colours, units, rows = self.contents()
        path = self.target()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_database(path, self.attributes, units, names, [lb or None for lb in labels], colours,
                       [[float(v) for v in row] for row in rows])
        network = os.path.splitext(path)[0] + ".nt"
        if not self.copy and self.numbers_changed(units, rows) and os.path.isfile(network):
            previous = os.path.splitext(path)[0] + ".previous.nt"
            os.replace(network, previous)
            self.network_set_aside = previous
        self.saved_path = path
        return path

    def accept(self):
        if self.check():
            return
        try:
            self.save()
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, "Database", f"Not saved: {e}")
            return
        super().accept()
