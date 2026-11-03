"""
materials_panel.py
==================
The Materials panel of the LSR MMTO window: the material database and its network.
Like every panel, a dialog with OK and Cancel working on a copy of the project: OK hands back
the edited parts (apply), Cancel leaves the project as it was. A value left as it was is handed
back exactly as it was, so a panel closed with OK and no edits changes nothing in the file.
"""
import math
import os
import subprocess
import sys

from PyQt5 import QtCore, QtWidgets

from lsr_mmto.problem.settings import BY_KEY, VAE_PARAMS as VAE_KEYS

from .panel_widgets import Info, database_choice, ok_cancel, same, swatch


def effective_text(d_eff, n_z):
    """d_eff with enough digits that rounding it up visibly gives n_z (1.004, not 1.00, for 2)."""
    for digits in (2, 3, 4, 6):
        text = f"{d_eff:.{digits}f}"
        if max(1, math.ceil(float(text))) == n_z:
            return text
    return f"{d_eff:.6g}"


def network_help(rows):
    """Every material as the trained network gives it back, as a table (rows from
    MaterialEncoder.reconstruction)."""
    from lsr_mmto.materials.databases import SHOWN
    head = ("<p>Each material is turned into its point on the material map and back by the network. The "
            "error is the difference in % of the property's range over the database's materials.</p>")
    cells = "".join(
        f"<tr><td>{r['material']}</td><td>{SHOWN.get(r['property'], r['property'])}</td>"
        f"<td align='right'>{r['database']:.6g}</td><td align='right'>{r['network']:.6g}</td>"
        f"<td align='right'>{r['difference']:+.3g}</td><td align='right'>{r['percent_of_range']:.2g}%</td></tr>"
        for r in rows)
    return (head + "<table cellspacing='0' cellpadding='2'><tr><th align='left'>Material</th>"
            "<th align='left'>Property</th><th>Database</th><th>Network</th><th>Difference</th><th>Error</th></tr>"
            + cells + "</table>")


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
        self.network_info = Info("")                                # every material as the network gives it back
        self.network_info.setVisible(False)
        below.addWidget(self.network_info, alignment=QtCore.Qt.AlignTop)
        layout.addLayout(below)
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
        self.network_info.setVisible(False)
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
            network = self._network_line(xlsx)
            if network:
                lines.insert(1, network)
        self.report.setText("\n".join(lines))

    def _network_line(self, xlsx):
        """How closely the trained network gives back the database's materials, in one line, with every
        value behind its (i); None when the database has no trained network yet."""
        from lsr_mmto.materials.encoder import network_reconstruction
        nt = self.project.network_path()
        if not (os.path.isfile(nt) and os.path.getsize(nt) > 0):
            return None
        try:
            rows = network_reconstruction(xlsx, nt)
        except Exception:                                           # noqa: BLE001
            return None
        if not rows:
            return None
        self.network_info.setToolTip(network_help(rows))
        self.network_info.setVisible(True)
        worst = max(r["percent_of_range"] for r in rows)
        return f"The trained network gives back every material within {worst:.2g}% of each property's range."

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
