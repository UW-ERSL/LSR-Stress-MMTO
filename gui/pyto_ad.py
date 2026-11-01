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
    FILE_DIALOG_OPTIONS

Also used, as MainWindow's own methods: open_structural_loads_window (picking), update_LivVar
(told when a pick is applied), and the Structural Loads window's redraw_loads and
refresh_applied_list.
"""
import PyTOImports  # noqa: F401  # type: ignore  # puts PyTO-AD on sys.path
from pyto.gui.PyTOGUI import (FILE_DIALOG_OPTIONS, GeometryWindow, MainWindow,  # noqa: F401  # type: ignore
                              ProjectsWindow, UnitsWindow)

UNIT_SYSTEMS = ("MKS", "mmKS", "IPS")          # the choices of UnitsWindow
