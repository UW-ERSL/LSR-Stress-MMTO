"""
pyto_ad_imports.py
==================
Every name the LSR MMTO GUI takes from the PyTO-AD GUI, in one place. A new PyTO-AD pin that
renames or removes one of them fails here, and in tests/test_gui.py, rather than deep inside
the GUI.

    MainWindow         the window: viewer, sidebar (BUTTON_CONFIG), message panel
    UnitsWindow        the unit system dialog, editing MainWindow.settings
    ProjectsWindow     its load_geometry draws an STL and turns picking on
    STLGeom            an STL read in: its triangles, normals, areas and neighbours
    MATERIAL_COLOR     the colour a part is drawn in
    StructuralLoadsWindow   picking supports and loads with the mouse
    DisplayOptionsWindow    cuts through the part, its triangles, bounding box and see-through view
    StructuralTopOptWindow  running an optimization: its worker thread, signals, live plot and log
    FILE_DIALOG_OPTIONS

Also used, as MainWindow's own methods: update_LivVar (told when a pick is applied) and
_raise_if_open; and the Structural Loads window's redraw_loads and refresh_applied_list.
"""
import lsr_mmto.pyto_ad_imports  # noqa: F401  # puts PyTO-AD on sys.path
from pyto.gui.PyTOGUI import (FILE_DIALOG_OPTIONS, MATERIAL_COLOR, DisplayOptionsWindow,  # noqa: F401
                              MainWindow, ProjectsWindow, StructuralLoadsWindow, StructuralTopOptWindow,
                              UnitsWindow)
from pyto.io.stl_reader import STLGeom  # noqa: F401  # type: ignore

UNIT_SYSTEMS = ("MKS", "mmKS", "IPS")          # the choices of UnitsWindow
