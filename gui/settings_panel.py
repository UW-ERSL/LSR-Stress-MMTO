"""
settings_panel.py
=================
The Settings panel of the LSR MMTO window: every setting, in groups.
OK and Cancel work as in every panel (see materials_panel.py).
"""
import os

from PyQt5 import QtCore, QtWidgets

from lsr_mmto.problem.settings import BY_KEY, MESH_DEFAULTS, SETTINGS, VAE_PARAMS as VAE_KEYS

from .panel_widgets import GREY, Info, NumberEdit, database_choice, database_errors, ok_cancel, same, with_info


def setting_groups():
    """(tab, group, [(section, key), ...]) in the order of SETTINGS; section "mesh" is Project.n_dof and
    Project.single_layer."""
    out = []
    for e in SETTINGS:
        if not out or out[-1][:2] != (e.tab, e.group):
            out.append((e.tab, e.group, []))
        out[-1][2].append((e.section, e.key))
    return out


VAE_NOTE = ("A database's network is the .nt file next to its .xlsx, with the same name (my_db.xlsx and "
            "my_db.nt). If there is none, the first run trains one. A database that comes with LSR MMTO can't be "
            "retrained: copy its .xlsx to a folder of your own first.")


def shown_setting(key, value):
    """A setting's value in words: on/off, a number, or what an empty one means."""
    if value is None:
        return BY_KEY[key].none_means or "none"
    if isinstance(value, bool):
        return "on" if value else "off"
    return format(value, "g") if isinstance(value, (int, float)) else str(value)


def setting_choices():
    """The settings with a fixed list of values."""
    return {e.key: list(e.choices.__members__) for e in SETTINGS if e.choices is not None}


def defaults_for(project):
    """What each setting is in a new project with this database: DEFAULT_SETTINGS, and for the VAE
    the database's trained values or the recommended latent dimension."""
    q = project.copy()
    q.settings = {}
    q.complete_settings()
    out = {s: dict(v) for s, v in q.settings.items()}
    out["mesh"] = dict(MESH_DEFAULTS)
    return out


class SettingRow:
    """One setting: its widget, and reading it back."""

    def __init__(self, key, value, choices):
        self.key = key
        self.none = None
        if BY_KEY[key].none_means:
            self.none = QtWidgets.QCheckBox(BY_KEY[key].none_means)
        if key in choices:
            self.widget = QtWidgets.QComboBox()
            self.widget.addItems(choices[key])
        elif key == "saveNet":
            self.widget = QtWidgets.QLineEdit()
        elif isinstance(value, bool) or key in ("single_layer",):
            self.widget = QtWidgets.QCheckBox()
        else:
            self.widget = NumberEdit(0, integer=BY_KEY[key].integer)
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
        for tab, group, keys in setting_groups():
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
                label = QtWidgets.QLabel(BY_KEY[key].label)
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
                grid.addWidget(Info(BY_KEY[key].help), r, 3)
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
        self.database_ok = database_choice(project)[1] is None and not database_errors(project)
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
        return [f"{BY_KEY[k].label} is not a number" for (s, k), entry in self.rows.items()
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
            changed.append(f"{BY_KEY[key].label} {shown_setting(key, old)} -> {shown_setting(key, new)}")
            if section == "mesh":
                setattr(project, key, int(new) if key == "n_dof" else bool(new))
            else:
                project.settings.setdefault(section, {})[key] = new
        out = ["Settings: " + ", ".join(changed)] if changed else []
        if self.vae_synced:
            out.append(f"{', '.join(BY_KEY[k].label for k in dict.fromkeys(self.vae_synced))}: set to the trained "
                       f"network's")
        return out


def all_setting_keys():
    """(section, key) of every setting the panel shows."""
    return [sk for _, _, keys in setting_groups() for sk in keys]
