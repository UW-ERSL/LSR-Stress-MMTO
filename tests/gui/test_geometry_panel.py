"""
test_geometry_panel.py
======================
The Geometry step (gui/geometry_panel.py): a new project has no part, Supports and loads is greyed and
Geometry opens; the steps made in the strip are the steps a script makes with the same calls; the part is built
and drawn only on Update part (and on closing the strip), the button and a line saying the view waits until
then; a step that cannot be made is said in red and the view keeps the last part; making a part of an STL
from shapes, or choosing an STL for a part of shapes, asks first; Paste points takes two columns of numbers
and names a row that is not.

    python -m pytest tests/gui/test_geometry_panel.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests", "gui"))

from PyQt5 import QtWidgets  # noqa: E402

from test_gui import Box, answers, scratch  # noqa: E402

from gui import geometry_panel as GP  # noqa: E402
from gui import main_window as G  # noqa: E402
from lsr_mmto import shapes as S  # noqa: E402
from lsr_mmto.problem.project import Project  # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
STL = os.path.join(ROOT, "data", "models", "LBracket_BM1", "LBracket_BM1.STL")


def typed(fields, key, text):
    """text typed in a box of fields and the box left, as a user does."""
    fields.boxes[key].setText(text)
    fields.boxes[key].editingFinished.emit()


def test_a_new_project_starts_with_no_part():
    """New project: no part, Supports and loads greyed saying why, Geometry open, the check saying so."""
    w = G.MMTOWindow()
    with scratch() as d:
        assert w.new_project(folder=os.path.join(d, "blank"))
        assert not w.project.has_part() and w.project.stl is None and w.project.part_shapes is None
        assert not w.sidebar_buttons["Supports and loads"].isEnabled()
        assert w.sidebar_buttons["Supports and loads"].toolTip() == G.NO_PART
        assert G.in_open_strip(w.geometry_panel)                          # Geometry opens by itself
        handler = next(c["handler"] for c in w.BUTTON_CONFIG if c["name"] == "Geometry")
        assert getattr(G.MMTOWindow, handler).__qualname__.startswith("MMTOWindow.")   # never PyTO-AD's
        assert getattr(w, handler)() is w.geometry_panel                  # own Load STL window: this strip
        assert any(e.startswith("no part yet") for e in w.project.validate(False))
        assert "No part yet" in w.message_text.toPlainText()
        assert Project.load(w.path).stl is None
    w.close()


def test_steps_made_in_the_strip_are_the_scripts_and_wait_for_update_part():
    """Steps typed in the strip equal the script's calls; drawn on Update part only; a step that cannot be made
    is said in red and the last part kept; closing builds; the file saves the steps."""
    w = G.MMTOWindow()
    with scratch() as d:
        assert w.new_project(folder=os.path.join(d, "plate"))
        g = w.geometry_panel
        g.side.buttons["shapes"].click()
        g.add_step("box")
        editor = g.editor_area.widget()
        typed(editor.shape_editor.fields, "x", "100")
        typed(editor.shape_editor.fields, "y", "40")
        typed(editor.shape_editor.fields, "z", "10")
        g.new_step()
        g.new_action.set("cut")
        g.add_step("cylinder")
        editor = g.editor_area.widget()
        typed(editor.shape_editor.fields, "radius", "5")
        typed(editor.place, "x", "20")
        typed(editor.place, "y", "20")
        editor._set_repeat("along_line")
        editor = g.editor_area.widget()
        typed(editor.repeat_fields, "count", "3")
        typed(editor.repeat_fields, "dx", "30")
        assert w.project.part_shapes == S.plain([S.add(S.box(100, 40, 10)),
                                                 S.cut(S.cylinder(5, 10), at=(20, 20, 0),
                                                       repeat=S.along_line(3, (30, 0, 0)))])
        assert g.steps_list.item(1).text().startswith("Cut cylinder r 5")
        assert w.part_waiting and GP.WAITING in g.message.text() and w.is_dirty()
        assert g.update_button.property("waiting") == "true"
        assert w.part_bounds() is None                                    # nothing drawn yet
        assert g.update_part() and not w.part_waiting
        lo, hi = w.part_bounds()
        assert list(hi - lo) == [100, 40, 10]
        assert os.path.normcase(w.stl_geom.file_path) == os.path.normcase(w.project.stl_path())
        assert w.sidebar_buttons["Supports and loads"].isEnabled()
        assert GP.WAITING not in g.message.text() and g.update_button.property("waiting") == "false"

        typed(g.editor_area.widget().shape_editor.fields, "radius", "0")       # a step that cannot be made
        assert not g.update_part()
        assert g.message.text().startswith("The part cannot be made: step 2:")
        assert list(w.part_bounds()[1] - w.part_bounds()[0]) == [100, 40, 10]   # the last part, kept
        typed(g.editor_area.widget().shape_editor.fields, "radius", "5")
        g.close()                                                         # closing updates the part
        assert not w.part_waiting and not [e for e in w.project.validate(False) if e.gui_step == "part"]
        assert w.save_project() and Project.load(w.path).part_shapes == w.project.part_shapes
    w.close()


def test_switching_between_an_stl_and_shapes_asks_first():
    """An STL part made from shapes, or a part of shapes given an STL, only after asking."""
    w = G.MMTOWindow()
    with scratch() as d:
        assert w.new_project(STL, os.path.join(d, "bracket"))
        g = w.open_geometry_panel()
        assert g.side.value() == "stl" and "LBracket_BM1.STL" in g.stl_name.text()
        g.side.buttons["shapes"].click()
        with answers(Box.No):
            assert g.add_step("sphere") is None                           # the STL stays
        assert w.project.stl is not None and not w.project.part_shapes
        with answers(Box.Yes):
            assert g.add_step("sphere")
        assert w.project.stl is None and len(w.project.part_shapes) == 1
        with answers(Box.No):
            assert not w.replace_part(STL)                                # the steps stay
        assert w.project.part_shapes
        with answers(Box.Yes):
            assert w.replace_part(STL)
        assert w.project.part_shapes is None and w.project.stl == "part/LBracket_BM1.STL"
    w.close()


def test_paste_points():
    """Paste points: two columns of numbers, a header skipped; a row that is not two numbers named."""
    editor = GP.OutlineEditor(S.polygon([(0, 0), (1, 0), (1, 1)]), "mm")
    assert editor.paste_points("x\ty\n0\t0\n10\t0\n10\t5\n0\t5\n")
    assert editor.outline()["points"] == [[0, 0], [10, 0], [10, 5], [0, 5]]
    assert not editor.paste_points("0 0\n1 a\n2 2\n")
    assert editor.error.text() == "Not pasted: row 2 ('1 a') is not two numbers"
    assert editor.outline()["points"] == [[0, 0], [10, 0], [10, 5], [0, 5]]         # nothing changed
    assert GP.points_from_text("0,0\n2,0\n1,3") == [[0, 0], [2, 0], [1, 3]]


def test_slips_that_could_lose_work():
    """A box with other text is not taken and crashes nothing; a choice clicked again changes nothing; the first
    step stays an Add; deleting every step leaves no part and no waiting; an example asks to be saved as a project
    of one's own before a step goes in; a wrap's boxes are those of the shape chosen only."""
    w = G.MMTOWindow()
    with scratch() as d:
        assert w.new_project(folder=os.path.join(d, "slips"))
        g = w.geometry_panel
        g.side.buttons["shapes"].click()
        g.add_step("box")
        editor = g.editor_area.widget()
        typed(editor.shape_editor.fields, "x", "abc")
        typed(editor.shape_editor.fields, "y", "20")                        # x still red: nothing taken
        assert w.project.part_shapes[0]["shape"]["size"] == [10, 10, 10]
        typed(editor.shape_editor.fields, "x", "2*5")
        assert w.project.part_shapes[0]["shape"]["size"] == [10, 20, 10]
        g.new_step()
        g.new_action.set("cut")
        g.add_step("extrusion")
        editor = g.editor_area.widget()
        editor._set_repeat("along_line")
        editor = g.editor_area.widget()
        typed(editor.repeat_fields, "dy", "3")
        editor.repeat_kind.buttons["along_line"].click()                    # already chosen
        typed(editor.shape_editor.outline_editor.fields, "depth", "3")
        editor.shape_editor.outline_editor.kind.buttons["rectangle"].click()
        step = w.project.part_shapes[1]
        assert step["repeat"]["step"][1] == 3 and step["shape"]["outline"]["depth"] == 3
        g.steps_list.setCurrentRow(1)
        g.move_step(-1)                                                     # a Cut first: refused
        assert [s["action"] for s in w.project.part_shapes] == ["add", "cut"]
        assert g.message.text().startswith("Not done: the first step must add")
        g.new_step()
        g.add_step("wrap")
        wrap = g.editor_area.widget().shape_editor
        def shown():                                                        # the boxes on screen, once the
            APP.processEvents()                                             # window has drawn its changes
            return sum(b.isVisibleTo(wrap) for b in wrap.findChildren(GP.NumberEdit))
        before = shown()
        wrap._add_member()                                                  # a cylinder, as the one shown
        assert shown() == before                                            # no boxes of the last shape left
        for _ in range(3):
            g.steps_list.setCurrentRow(g.steps_list.count() - 1)
            g.delete_step()
        g.update_part()
        assert w.project.part_shapes == [] and not w.project.has_part() and not w.part_waiting
        assert not w.sidebar_buttons["Supports and loads"].isEnabled() and w.part_bounds() is None
        with answers(Box.Discard):
            assert w.open_project(os.path.join(ROOT, "projects", "U-bracket.mmto"))
        g = w.open_geometry_panel()
        g.side.buttons["shapes"].click()
        with answers(Box.Cancel):
            assert g.add_step("box") is None                                # an example is never written
        assert w.project.part_shapes is None and w.template
    w.close()
