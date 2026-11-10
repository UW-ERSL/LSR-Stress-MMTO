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

from .panel_widgets import GREY, Info, fit_columns, ok_cancel

# the properties a new database starts with; Add property offers every property of the registry
# (databases.ATTRIBUTES)
STARTING = ["Density", "Youngs_Modulus", "Poissons_Ratio"]
FIXED = ["Material", "Shown as", "Colour"]                  # the columns before the properties
FIXED_WIDTHS = [160, 100, 80]                               # their least widths
BAD = QtGui.QColor("#ffd6d6")
HEADER_SHADE = "#f3f5f8"                                    # the headers' colour (app_style.qss), the units row's
NO_UNIT = "no unit"                                         # the units row of a property without units
UNIT_MARGIN = 3                                             # around a unit's drop-down in the units row
HELP = ("One row per material. Shown as is optional: a shorter name for legends and plots (the name is "
        "shown when it is empty). Click a colour to change it.\n\nThe row under the headers holds each "
        "property's unit. Add property opens the list of properties to add, a column each; Remove property removes "
        "the column you are in.\n\n"
        "You can paste a block copied from Excel into the table.")


def shown(key):
    """A property's name as the user reads it (databases.SHOWN), else its own name."""
    from lsr_mmto.materials.databases import SHOWN
    return SHOWN.get(key, key)


class CentredCombo(QtWidgets.QComboBox):
    """A drop-down whose chosen text is drawn centred in the whole box."""

    def paintEvent(self, event):
        painter = QtWidgets.QStylePainter(self)
        option = QtWidgets.QStyleOptionComboBox()
        self.initStyleOption(option)
        text, option.currentText = option.currentText, ""
        painter.drawComplexControl(QtWidgets.QStyle.CC_ComboBox, option)
        painter.drawText(self.rect(), QtCore.Qt.AlignCenter, text)


class FrozenFirstColumn(QtWidgets.QTableView):
    """The table's first column kept in place over its left edge while the table scrolls sideways, as Excel's
    freeze panes: a second view of the same cells (the same model and selection), showing only that column.
    A paste or Delete in it goes to the table."""

    def __init__(self, table):
        super().__init__(table)
        self.table = table
        self.setObjectName("frozenColumn")
        self.setModel(table.model())
        self.setSelectionModel(table.selectionModel())
        self.setEditTriggers(table.editTriggers())
        self.verticalHeader().hide()
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        for bar in (self.setHorizontalScrollBarPolicy, self.setVerticalScrollBarPolicy):
            bar(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollMode(table.verticalScrollMode())
        self.setStyleSheet("QTableView#frozenColumn { border: none; }")
        table.viewport().stackUnder(self)
        table.verticalScrollBar().valueChanged.connect(self.verticalScrollBar().setValue)
        self.verticalScrollBar().valueChanged.connect(table.verticalScrollBar().setValue)
        table.horizontalHeader().sectionResized.connect(lambda c, _, new: self._width(c, new))
        table.verticalHeader().sectionResized.connect(lambda r, _, new: self.setRowHeight(r, new))
        for signal in (table.model().columnsInserted, table.model().columnsRemoved, table.model().rowsInserted):
            signal.connect(lambda *_: self.follow())
        self.follow()

    def _width(self, c, width):
        if c == 0:
            self.setColumnWidth(0, width)
            self.place()

    def follow(self):
        """Only the first column, as wide and as tall as the table's, over its left edge."""
        for c in range(self.model().columnCount()):
            self.setColumnHidden(c, c != 0)
        self.setColumnWidth(0, self.table.columnWidth(0))
        for r in range(self.model().rowCount()):
            self.setRowHeight(r, self.table.rowHeight(r))
        self.horizontalHeader().setFixedHeight(self.table.horizontalHeader().height())
        self.place()

    def place(self):
        t = self.table
        self.setGeometry(t.verticalHeader().width() + t.frameWidth(), t.frameWidth(), t.columnWidth(0),
                         t.viewport().height() + t.horizontalHeader().height())

    def keyPressEvent(self, event):
        if event.matches(QtGui.QKeySequence.Paste) or event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            self.table.keyPressEvent(event)
            return
        super().keyPressEvent(event)
        if self.state() == self.EditingState:       # a key that started typing in a cell here: not the table's too,
            event.accept()                          # which would type it a second time (B became BB)


class _Table(QtWidgets.QTableWidget):
    """A table that takes a block pasted from Excel (tab between cells, a new line between rows), adding
    material rows as needed. Its first column (Material) stays in place when it scrolls sideways."""
    pasted = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.frozen = FrozenFirstColumn(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.frozen.follow()

    def updateGeometries(self):
        super().updateGeometries()
        if hasattr(self, "frozen"):
            self.frozen.place()

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


class AddPropertiesDialog(QtWidgets.QDialog):
    """Add property: every property LSR MMTO knows (databases.ATTRIBUTES), each with a tick box, found by
    typing in the search box. With a project, in two parts: those the project needs (Project.needed_properties)
    and the others, each part in alphabetical order; without one, all in one list. Those already in the table
    are ticked and greyed. OK says how many it adds; chosen() gives them."""

    def __init__(self, parent, in_table, project=None):
        from lsr_mmto.materials.databases import ATTRIBUTES
        super().__init__(parent)
        self.setWindowTitle("Add properties")
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.setFixedSize(460, 520)
        layout = QtWidgets.QVBoxLayout(self)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search properties")
        self.search.textChanged.connect(self._filter)
        layout.addWidget(self.search)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(project is not None)
        layout.addWidget(self.tree, stretch=1)
        needed = [k for k in project.needed_properties() if k in ATTRIBUTES] if project is not None else []
        others = [k for k in ATTRIBUTES if k not in needed]
        by_shown = lambda keys: sorted(keys, key=lambda k: shown(k).lower())        # noqa: E731
        groups = ([("Needed by this project", by_shown(needed)), ("Other properties", by_shown(others))]
                  if project is not None else [(None, by_shown(others))])
        self.items = {}
        for title, keys in groups:
            parent_item = self.tree.invisibleRootItem()
            if title is not None:
                parent_item = QtWidgets.QTreeWidgetItem(self.tree, [title])
                parent_item.setFlags(QtCore.Qt.ItemIsEnabled)
                font = parent_item.font(0)
                font.setBold(True)
                parent_item.setFont(0, font)
                parent_item.setExpanded(True)
            for key in keys:
                there = key in in_table
                item = QtWidgets.QTreeWidgetItem(parent_item, [shown(key) + ("    (in the table)" if there else "")])
                item.setData(0, QtCore.Qt.UserRole, key)
                item.setFlags(QtCore.Qt.ItemIsUserCheckable | (QtCore.Qt.NoItemFlags if there else
                                                                QtCore.Qt.ItemIsEnabled))
                item.setCheckState(0, QtCore.Qt.Checked if there else QtCore.Qt.Unchecked)
                if there:
                    item.setForeground(0, QtGui.QColor("#9aa1ab"))                      # greyed
                self.items[key] = item
        self.tree.expandAll()
        self.tree.itemChanged.connect(lambda *_: self._count())
        note = QtWidgets.QLabel("Properties already in the table are ticked. Tick the ones to add.")
        note.setStyleSheet(GREY)
        note.setWordWrap(True)
        layout.addWidget(note)
        self.in_table = set(in_table)
        self.buttons = ok_cancel(self, layout)
        self.add_button = self.buttons.button(QtWidgets.QDialogButtonBox.Ok)
        self._count()

    def groups(self):
        """{part's title (None without parts): the keys listed in it, in order}."""
        out = {}
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            top = root.child(i)
            if top.data(0, QtCore.Qt.UserRole) is not None:
                out.setdefault(None, []).append(top.data(0, QtCore.Qt.UserRole))
            else:
                out[top.text(0)] = [top.child(j).data(0, QtCore.Qt.UserRole) for j in range(top.childCount())]
        return out

    def chosen(self):
        """The keys ticked to add (not those already in the table), in the order listed."""
        return [k for k, item in self.items.items() if k not in self.in_table and
                item.checkState(0) == QtCore.Qt.Checked]

    def _count(self):
        n = len(self.chosen())
        self.add_button.setText(f"Add {n} propert{'y' if n == 1 else 'ies'}" if n else "Add")
        self.add_button.setEnabled(bool(n))

    def _filter(self, text):
        """Only the properties whose name (any of them) holds the text typed; a part with none is hidden."""
        from lsr_mmto.materials.databases import ATTRIBUTES
        text = text.strip().lower()
        for key, item in self.items.items():
            a = ATTRIBUTES[key]
            names = [a.shown, a.name] + list(a.aliases)
            item.setHidden(bool(text) and not any(text in n.lower() for n in names))
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            top = root.child(i)
            if top.childCount():
                top.setHidden(all(top.child(j).isHidden() for j in range(top.childCount())))

    def shown_keys(self):
        """The keys the list shows now (after the search)."""
        return [k for k, item in self.items.items() if not item.isHidden() and
                (item.parent() is None or not item.parent().isHidden())]


class DatabaseEditor(QtWidgets.QDialog):
    """New (path None: a database to be saved in folder) or an existing database to change. A shipped one
    (in_shipped_folder) opens as a copy to be saved in folder. imported (databases.import_table, with its
    file's name as "stem"): a new database filled from a table made elsewhere, to be checked and saved.
    saved_path is set by a successful Save, and network_set_aside names the network moved out of the way
    when the numbers changed."""

    def __init__(self, parent, folder, path=None, unit_system=None, imported=None, project=None):
        from lsr_mmto.materials.databases import in_shipped_folder
        super().__init__(parent)
        path = None if path is None else os.path.normpath(os.path.abspath(path))   # one kind of slash on Windows
        self.folder, self.path, self.unit_system, self.project = folder, path, unit_system, project
        self.copy = path is not None and in_shipped_folder(path)
        self.saved_path, self.network_set_aside, self.loaded = None, None, None
        self.left_out = []                                          # the file's columns of no known property
        self.setWindowTitle("New database" if path is None else "Database")
        self.resize(900, 520)
        layout = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Database name"))
        self.name = QtWidgets.QLineEdit()
        top.addWidget(self.name, stretch=1)
        self.where = QtWidgets.QLabel()
        self.where.setStyleSheet(GREY)
        top.addWidget(self.where, stretch=2)
        top.addWidget(Info(HELP))
        layout.addLayout(top)
        self.left_out_note = QtWidgets.QLabel()                     # columns of the file it does not know
        self.left_out_note.setWordWrap(True)
        self.left_out_note.setStyleSheet(GREY)
        self.left_out_note.setVisible(False)
        layout.addWidget(self.left_out_note)

        self.table = _Table()
        self.table.parent_editor = self
        self.table.verticalHeader().setVisible(False)
        self.table.cellClicked.connect(self._clicked)
        self.table.itemChanged.connect(self._edited)
        self.table.pasted.connect(self.check)
        self.touched = set()                                        # the rows edited: only they are marked
        self.unit_boxes = {}                                        # property -> its unit's drop-down
        layout.addWidget(self.table, stretch=1)

        rows = QtWidgets.QHBoxLayout()                              # four of one width, the table's width
        self.add_property_button = QtWidgets.QPushButton("Add property")
        self.row_buttons = [QtWidgets.QPushButton(text) for text in ("Add material", "Remove material")] + \
            [self.add_property_button, QtWidgets.QPushButton("Remove property")]
        for button, slot in zip(self.row_buttons, (self.add_material, self.remove_material, self.add_properties,
                                                   self.remove_property)):
            button.clicked.connect(lambda _=False, s=slot: s())
            button.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
            rows.addWidget(button, stretch=1)
        layout.addLayout(rows)
        self.problems_text = QtWidgets.QLabel()
        self.problems_text.setWordWrap(True)
        self.problems_text.setStyleSheet("color: #b00000;")
        layout.addWidget(self.problems_text)
        self.buttons = ok_cancel(self, layout)
        self.save_button = self.buttons.button(QtWidgets.QDialogButtonBox.Ok)
        self.save_button.setText("Save")

        self.attributes = []
        if imported is not None:
            self._fill(imported)
        elif path is None:
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
        """The database at path in the table. Columns of no property LSR MMTO knows are left out and named
        in a note over the table, so that Save writes the database without them."""
        from lsr_mmto.materials.databases import ATTRIBUTES, database_table, material_display
        names, attributes, units, values = database_table(path)
        colours, labels = material_display(path)
        known = [j for j, key in enumerate(attributes) if key in ATTRIBUTES]
        self.left_out = [str(key).strip() for key in attributes if key not in ATTRIBUTES]
        attributes, units, values = [attributes[j] for j in known], [units[j] for j in known], values[:, known]
        if self.left_out:
            self.left_out_note.setText(f"Left out, as LSR MMTO has no property of that name: "
                                       f"{', '.join(self.left_out)}. Save writes the database without them.")
            self.left_out_note.setVisible(True)
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

    def _fill(self, imported):
        """A new database from a table made elsewhere: its materials, properties, units and values as read,
        each value as text so that one that is not a number shows as a cell to correct."""
        self.setWindowTitle("Imported database")
        self.name.setText(self._free_name(imported["stem"]))
        self.table.blockSignals(True)
        self._columns(order=imported["attributes"], units=imported["units"])
        for name, row in zip(imported["names"], imported["rows"]):
            r = self._new_row()
            self.table.item(r, 0).setText(name)
            for j, v in enumerate(row):
                self.table.item(r, 3 + j).setText(v)
        self.table.blockSignals(False)

    # ------------------------------------------------------------------ the table
    def _default_unit(self, key):
        from lsr_mmto.materials.databases import ATTRIBUTES, UNIT_SYSTEMS
        a = ATTRIBUTES[key]
        return UNIT_SYSTEMS.get(self.unit_system or "", {}).get(a.measures, a.units[0])

    def add_properties(self):
        """Add property: the Add properties window; the properties ticked there get a column each, at the end,
        empty. Returns them."""
        dialog = AddPropertiesDialog(self, self.attributes, self.project)
        if not dialog.exec_():
            return []
        keys = dialog.chosen()
        self.add_property(*keys)
        return keys

    def add_property(self, *keys):
        """A column for each property not in the table yet, at the end, empty."""
        new = [k for k in keys if k not in self.attributes]
        if new:
            self._columns(order=self.attributes + new)

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
                self, "Remove property", f"Remove {shown(key)} and its numbers?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            return False
        self._columns(order=[k for k in self.attributes if k != key])
        return True

    def _columns(self, *_, order=None, units=None):
        """The property columns, in order (by default as they are); a column that stays keeps its numbers
        and unit. Row 0 holds the units, in the headers' band: a drop-down for each property with units,
        "no unit" for one without."""
        from lsr_mmto.materials.databases import ATTRIBUTES
        keep = {}
        for j, key in enumerate(self.attributes):
            keep[key] = (self.unit(key), [self.table.item(r, 3 + j).text() for r in range(1, self.table.rowCount())])
        wanted = list(order) if order is not None else list(self.attributes)
        was = self.table.blockSignals(True)
        self.table.setColumnCount(len(FIXED) + len(wanted))
        self.table.setHorizontalHeaderLabels(FIXED + [shown(k) for k in wanted])
        if self.table.rowCount() == 0:
            self.table.setRowCount(1)
        for c in range(len(FIXED)):
            item = QtWidgets.QTableWidgetItem("")
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            item.setBackground(QtGui.QColor(HEADER_SHADE))
            self.table.setItem(0, c, item)
        self.unit_boxes = {}
        for j, key in enumerate(wanted):
            a = ATTRIBUTES[key]
            unit = (units[j] if units else None) or keep.get(key, (None,))[0] or self._default_unit(key)
            item = QtWidgets.QTableWidgetItem("")
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            item.setBackground(QtGui.QColor(HEADER_SHADE))
            self.table.setItem(0, 3 + j, item)
            self.table.setCellWidget(0, 3 + j, self._unit_cell(key, a, unit))
            old = keep.get(key, (None, []))[1]
            for r in range(1, self.table.rowCount()):
                self.table.setItem(r, 3 + j, QtWidgets.QTableWidgetItem(old[r - 1] if r - 1 < len(old) else ""))
        self.attributes = list(wanted)
        self.table.blockSignals(was)
        self.table.setRowHeight(0, self.table.verticalHeader().defaultSectionSize())   # as tall as a material row
        fit_columns(self.table, least=FIXED_WIDTHS)
        self.table.frozen.follow()
        self.check()

    def _unit_cell(self, key, a, unit):
        """Row 0's cell of a property: its unit's drop-down, slim, with an even margin on the headers' colour;
        "no unit" in grey for a property without units."""
        holder = QtWidgets.QWidget()
        holder.setStyleSheet(f"background: {HEADER_SHADE};")
        box = QtWidgets.QHBoxLayout(holder)
        box.setContentsMargins(3 * UNIT_MARGIN, UNIT_MARGIN, 3 * UNIT_MARGIN, UNIT_MARGIN)
        if a.measures == "ratio":
            label = QtWidgets.QLabel(NO_UNIT)
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setStyleSheet("color: #6b7280;")
            box.addWidget(label)
            return holder
        combo = CentredCombo()
        combo.addItems(list(a.units))
        if combo.findText(unit) < 0:
            combo.addItem(unit)
        combo.setCurrentText(unit)
        combo.setStyleSheet("QComboBox { background: #ffffff; padding: 0px 6px; min-height: 0px; }")
        combo.setFixedHeight(self.table.verticalHeader().defaultSectionSize() - 2 * UNIT_MARGIN)
        combo.currentTextChanged.connect(lambda _: self.check())
        box.addWidget(combo)
        self.unit_boxes[key] = combo
        return holder

    def unit(self, key):
        """The unit chosen for a property in the units row; its one unit for a property without units."""
        from lsr_mmto.materials.databases import ATTRIBUTES
        box = self.unit_boxes.get(key)
        return box.currentText() if box is not None else (ATTRIBUTES[key].units[0] if key in ATTRIBUTES else None)

    def _new_row(self, colour=None):
        from lsr_mmto.materials.databases import default_colours
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
        units = [self.unit(key) for key in self.attributes]
        return ([cell(r, 0) for r in rows], [cell(r, 1) for r in rows], [self.colour(r) for r in rows], units,
                [[cell(r, 3 + j) for j in range(len(self.attributes))] for r in rows])

    def target(self):
        if self.path is not None and not self.copy:
            return self.path
        return os.path.join(self.folder, self.name.text().strip() + ".xlsx")

    def _where(self):
        """The grey words beside the name: where Save writes, the whole path in its tip."""
        from lsr_mmto.problem.project_folder import MATERIALS
        folder = os.path.dirname(self.target())
        self.where.setText("Saved in the project's materials folder" if os.path.basename(folder) == MATERIALS
                           else f"Saved in {folder}")
        self.where.setToolTip(self.target())

    def _edited(self, item):
        """A cell typed in: its row is marked from now on when something in it is wrong."""
        self.touched.add(item.row())
        self.check()

    def check(self):
        """Every cell checked; Save only with none. The problems. A bad cell is shaded, with the reason on it,
        and its problem said under the table, only in a row that was edited or holds something (a blank new
        form shows no red)."""
        from lsr_mmto.materials.databases import ATTRIBUTES
        if self.table.columnCount() < len(FIXED):
            return ["no columns"]
        names, labels, colours, units, rows = self.contents()
        out, said = [], []                                          # every problem; those said under the table
        was = self.table.blockSignals(True)
        for r, name in enumerate(names, start=1):
            seen = r in self.touched or bool(name or labels[r - 1] or any(rows[r - 1]))
            bad = not name or names.count(name) > 1
            self._mark(r, 0, bad and seen, "A name, different from the others" if bad else "")
            if bad and seen and not said:
                said.append("Every material needs a name of its own")
            for j, key in enumerate(self.attributes):
                a, text = ATTRIBUTES[key], rows[r - 1][j]
                try:
                    value = float(text)
                    reason = "" if a.in_range(value) else f"{shown(key)}: {a.range_text()}"
                except ValueError:
                    reason = f"{shown(key)}: a number" if text else f"{shown(key)}: empty"
                self._mark(r, 3 + j, bool(reason) and seen, reason)
                if reason:
                    out.append(f"{name or 'row ' + str(r)}: {reason}")
                    if seen:
                        said.append(out[-1])
        self.table.blockSignals(was)
        if any(not n or names.count(n) > 1 for n in names):
            out.insert(0, "Every material needs a name of its own")
        whole = []                                                  # about the whole database: always said
        if len(names) < 2:
            whole.append("At least two materials")
        if not self.attributes:
            whole.append("Add at least one property")
        file_name = self.name.text().strip()
        if (self.path is None or self.copy) and (not file_name or any(ch in file_name for ch in '\\/:*?"<>|')):
            whole.append("Name: letters, digits, spaces, - and _")
        elif (self.path is None or self.copy) and os.path.exists(self.target()):
            whole.append(f"{os.path.basename(self.target())} is already there; choose another name")
        out += whole
        said += whole
        self.problems_text.setText(said[0] + (f" (and {len(said) - 1} more)" if len(said) > 1 else "") if said else "")
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
        from lsr_mmto.materials.databases import write_database
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
