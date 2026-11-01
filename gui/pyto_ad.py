"""
pyto_ad.py
==========
Every name the LSR MMTO GUI takes from the PyTO-AD GUI, in one place. A new PyTO-AD pin that
renames or removes one of them fails here, and in tests/test_gui.py, rather than deep inside
the GUI.

    MainWindow         the window: viewer, sidebar (BUTTON_CONFIG), message panel
    UnitsWindow        the unit system dialog, editing MainWindow.settings
    GeometryWindow     loading an STL (also from SolidWorks)
    ProjectsWindow     its load_geometry draws an STL and turns picking on
    StructuralLoadsWindow   picking supports and loads with the mouse
    FILE_DIALOG_OPTIONS

Also used, as MainWindow's own methods: update_LivVar (told when a pick is applied) and
_raise_if_open; and the Structural Loads window's redraw_loads and refresh_applied_list.
"""
import PyTOImports  # noqa: F401  # type: ignore  # puts PyTO-AD on sys.path
from pyto.gui.PyTOGUI import (FILE_DIALOG_OPTIONS, GeometryWindow, MainWindow,  # noqa: F401  # type: ignore
                              ProjectsWindow, StructuralLoadsWindow, UnitsWindow)

UNIT_SYSTEMS = ("MKS", "mmKS", "IPS")          # the choices of UnitsWindow
