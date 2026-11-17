"""
database_versions_window.py
===========================
Versions... in the Materials panel: the versions of the project's database (lsr_mmto/materials/
database_versions.py), newest first, "(in use)" beside the one the database is now. Restore puts the chosen
one back as the database, without asking (every version stays, the one replaced too), and closes the window;
Rename... changes a version's note; Delete deletes one after a confirmation, never the one in use.
"""
import os
import time

from PyQt5 import QtWidgets

from . import database_editor                              # its ask: one place for the questions


def version_text(stem, version, in_use):
    """A version as the list shows it: its name ("steels v3 softer"), "(in use)" for the one in use, when its
    contents were saved, how many materials."""
    when = time.strftime("%d %b %Y, %H:%M", time.localtime(version.when))
    count = "cannot be read" if version.materials < 0 else \
        f"{version.materials} material{'' if version.materials == 1 else 's'}"
    return f"{stem} {version.name}{' (in use)' if in_use else ''}    {when}    {count}"


class VersionsWindow(QtWidgets.QDialog):
    """The versions of the database xlsx. restored is the version put back, None until one is."""

    def __init__(self, parent, xlsx):
        super().__init__(parent)
        self.xlsx, self.stem, self.restored = xlsx, os.path.splitext(os.path.basename(xlsx))[0], None
        self.setWindowTitle(f"Versions of {self.stem}")
        self.resize(520, 320)
        layout = QtWidgets.QVBoxLayout(self)
        self.list = QtWidgets.QListWidget()
        layout.addWidget(self.list, stretch=1)
        self.empty = QtWidgets.QLabel(f"No versions of {self.stem} yet.")
        layout.addWidget(self.empty)
        buttons = QtWidgets.QHBoxLayout()
        self.rename_button = QtWidgets.QPushButton("Rename...")
        buttons.addWidget(self.rename_button)
        buttons.addStretch()
        self.delete_button = QtWidgets.QPushButton("Delete")
        self.restore_button = QtWidgets.QPushButton("Restore")
        close = QtWidgets.QPushButton("Close")
        for button, slot in ((self.rename_button, self.rename), (self.delete_button, self.delete),
                             (self.restore_button, self.restore), (close, self.accept)):
            button.clicked.connect(lambda _=False, s=slot: s())
            if button is not self.rename_button:
                buttons.addWidget(button)
        layout.addLayout(buttons)
        self.list.currentRowChanged.connect(lambda _: self._buttons())
        self.fill()

    def fill(self, chosen=None):
        """The list read again; chosen (a version's number) chosen in it, else the newest."""
        from lsr_mmto.materials.database_versions import in_use, versions
        self.versions = versions(self.xlsx)
        current = in_use(self.xlsx, self.versions)
        self.current = None if current is None else current.path
        self.list.clear()
        for v in self.versions:
            self.list.addItem(version_text(self.stem, v, v.path == self.current))
        self.list.setVisible(bool(self.versions))
        self.empty.setVisible(not self.versions)
        if self.versions:
            rows = [i for i, v in enumerate(self.versions) if v.number == chosen]
            self.list.setCurrentRow(rows[0] if rows else 0)
        self._buttons()

    def _buttons(self):
        version = self.chosen()
        used = version is not None and version.path == self.current
        self.rename_button.setEnabled(version is not None)
        self.delete_button.setEnabled(version is not None and not used)
        self.restore_button.setEnabled(version is not None and not used and version.materials >= 0)
        self.delete_button.setToolTip("The version in use is the database as it is now: it can't be deleted."
                                      if used else "")

    def chosen(self):
        row = self.list.currentRow()
        return self.versions[row] if 0 <= row < len(self.versions) else None

    def restore(self):
        """The chosen version back as the database, without asking (the database as it is now is kept first if
        no version has it: a change made in Excel while this window was open, say); the window closes. True when
        done."""
        from lsr_mmto.materials import database_versions as V
        version = self.chosen()
        if version is None or version.path == self.current:
            return False
        try:
            V.restore(version, self.xlsx)
        except (OSError, ValueError) as e:
            QtWidgets.QMessageBox.warning(self, "Restore", f"Not restored: {e}. If the database is open in Excel, "
                                                           f"close it there and try again.")
            self.fill(version.number)
            return False
        self.restored = version
        self.accept()
        return True

    def rename(self, note=None):
        """The chosen version's note changed, asked for when note is None ("" takes it away). True when done."""
        from lsr_mmto.materials import database_versions as V
        version = self.chosen()
        if version is None:
            return False
        if note is None:
            note, ok = QtWidgets.QInputDialog.getText(
                self, "Rename version", f"A note after {self.stem} v{version.number} (empty for none):",
                text=version.note)
            if not ok:
                return False
        try:
            V.set_note(version, note)
        except (ValueError, OSError) as e:
            QtWidgets.QMessageBox.warning(self, "Rename version", f"Not renamed: {e}.")
            return False
        self.fill(version.number)
        return True

    def delete(self):
        """The chosen version deleted, after a yes; never the one in use. True when done."""
        from lsr_mmto.materials import database_versions as V
        version, Box = self.chosen(), QtWidgets.QMessageBox
        if version is None or version.path == self.current:
            return False
        text = f"Delete {self.stem} {version.name}? This can't be undone."
        if database_editor.ask(self, "Delete version", text, Box.Yes | Box.No, Box.No) != Box.Yes:
            return False
        try:
            V.delete(version, self.xlsx)
        except (OSError, ValueError) as e:
            QtWidgets.QMessageBox.warning(self, "Delete version", f"Not deleted: {e}.")
            self.fill()
            return False
        self.fill()
        return True
