"""
database_versions_window.py
===========================
Versions... in the Materials panel: the older versions kept of the project's database
(lsr_mmto/materials/database_versions.py), newest first. Restore puts one back in the database's place,
after asking whether to keep the current version first; Delete deletes one, after a confirmation.
"""
import os
import time

from PyQt5 import QtWidgets

from . import database_editor                              # its ask: one place for the questions


def version_text(version):
    """A version as the list shows it: when it was kept, whether it is the first, how many materials."""
    when = time.strftime("%d %b %Y, %H:%M:%S", time.localtime(version.when))
    first = ", first version" if version.first else ""
    count = "cannot be read" if version.materials < 0 else \
        f"{version.materials} material{'' if version.materials == 1 else 's'}"
    return f"{when}{first}    {count}"


class VersionsWindow(QtWidgets.QDialog):
    """The versions of the database xlsx. restored is True once one was put back."""

    def __init__(self, parent, xlsx):
        super().__init__(parent)
        self.xlsx, self.stem, self.restored = xlsx, os.path.splitext(os.path.basename(xlsx))[0], False
        self.setWindowTitle(f"Versions of {self.stem}")
        self.resize(460, 320)
        layout = QtWidgets.QVBoxLayout(self)
        self.list = QtWidgets.QListWidget()
        layout.addWidget(self.list, stretch=1)
        self.empty = QtWidgets.QLabel(f"No older versions of {self.stem} yet.")
        layout.addWidget(self.empty)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch()
        self.delete_button = QtWidgets.QPushButton("Delete")
        self.restore_button = QtWidgets.QPushButton("Restore")
        close = QtWidgets.QPushButton("Close")
        for button, slot in ((self.delete_button, self.delete), (self.restore_button, self.restore),
                             (close, self.accept)):
            button.clicked.connect(lambda _=False, s=slot: s())
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.list.currentRowChanged.connect(lambda _: self._buttons())
        self.fill()

    def fill(self):
        from lsr_mmto.materials.database_versions import versions
        self.versions = versions(self.xlsx)
        self.list.clear()
        for v in self.versions:
            self.list.addItem(version_text(v))
        self.list.setVisible(bool(self.versions))
        self.empty.setVisible(not self.versions)
        if self.versions:
            self.list.setCurrentRow(0)
        self._buttons()

    def _buttons(self):
        chosen = self.list.currentRow() >= 0 and bool(self.versions)
        self.delete_button.setEnabled(chosen)
        self.restore_button.setEnabled(chosen)

    def chosen(self):
        row = self.list.currentRow()
        return self.versions[row] if 0 <= row < len(self.versions) else None

    def restore(self):
        """The chosen version back as the database, after Yes or No to keeping the current one. True when done."""
        from lsr_mmto.materials import database_versions as V
        version, Box = self.chosen(), QtWidgets.QMessageBox
        if version is None:
            return False
        answer = database_editor.ask(self, f"Restore {self.stem}", "Keep the current version?", Box.Yes | Box.No | Box.Cancel,
                     Box.Yes)
        if answer not in (Box.Yes, Box.No):
            return False
        if answer == Box.Yes:
            V.keep(self.xlsx)
        V.restore(version, self.xlsx)
        self.restored = True
        self.fill()
        return True

    def delete(self):
        """The chosen version deleted, after a yes. True when done."""
        from lsr_mmto.materials import database_versions as V
        version, Box = self.chosen(), QtWidgets.QMessageBox
        if version is None:
            return False
        text = (f"This is the first version of {self.stem}. Delete it? This can't be undone." if version.first
                else f"Delete this version of {self.stem}? This can't be undone.")
        if database_editor.ask(self, "Delete version", text, Box.Yes | Box.No, Box.No) != Box.Yes:
            return False
        V.delete(version)
        self.fill()
        return True

