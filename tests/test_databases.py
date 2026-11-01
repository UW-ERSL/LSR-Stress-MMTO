"""
test_databases.py
=================
Material databases: the attribute registry, the check of a database file, the latent
dimension recommended for it, the Excel template, and what a Project does with them.
Small files written on the fly; a few seconds.

    python tests/test_databases.py        (or: python -m pytest tests)
"""
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils import project_paths  # noqa: E402

project_paths.bootstrap()

from openpyxl import Workbook, load_workbook  # noqa: E402
from databases import (ATTRIBUTES, DATABASE_DIR, DATABASE_VAE, VAEParams, attribute_name,  # noqa: E402
                       check_database, convert_database, database_units, default_colours,
                       material_display, recommended_latent_dim, unit_ratio, write_template)  # type: ignore

SHIPPED = os.path.join(ROOT, DATABASE_DIR)
GOOD = [["Attributes", "Density", "Youngs_Modulus", "Yield_Strength"],
        ["Material/Units", "kg/m^3", "Pa", "Pa"],
        ["A", 1.0, 100.0, 100.0],
        ["B", 1.2, 160.0, 140.0],
        ["C", 0.8, 70.0, 90.0]]


def write(path, rows, display=None):
    wb = Workbook()
    for r in rows:
        wb.active.append(r)
    if display is not None:
        ws = wb.create_sheet("Display")
        for r in display:
            ws.append(r)
    wb.save(path)
    return path


def errors_of(rows, display=None):
    with tempfile.TemporaryDirectory() as d:
        return check_database(write(os.path.join(d, "db.xlsx"), rows, display))


def test_shipped_databases():
    """The four databases pass with no errors and no warnings, and the recommended latent
    dimension is the one each network was trained with."""
    from materialEncoder import MaterialEncoder  # type: ignore
    import contextlib
    import io
    for name, vae in DATABASE_VAE.items():
        path = os.path.join(SHIPPED, f"{name}.xlsx")
        assert check_database(path) == ([], []), name
        d_eff, n_z = recommended_latent_dim(path)
        assert n_z == vae["latentDim"], (name, d_eff, n_z)
        enc = MaterialEncoder(VAEParams())
        with contextlib.redirect_stdout(io.StringIO()):
            enc.readExcel(path)
        assert abs(enc.intrinsic_dim - d_eff) < 1e-5, (name, enc.intrinsic_dim, d_eff)


def test_errors_name_the_cell():
    assert errors_of(GOOD) == ([], [])
    cases = [
        (lambda r: r[0].__setitem__(2, "Youngs_Modulous"), "C1: 'Youngs_Modulous' is not a known attribute"),
        (lambda r: r[0].__setitem__(3, "E"), "D1: Youngs_Modulus appears twice"),
        (lambda r: r[0].__setitem__(1, None), "B1: no attribute name"),
        (lambda r: r[3].__setitem__(2, -5.0), "C4 (B, Youngs_Modulus): -5 is outside the allowed range"),
        (lambda r: r[3].__setitem__(2, "stiff"), "C4 (B, Youngs_Modulus): 'stiff' is not a number"),
        (lambda r: r[4].__setitem__(1, None), "B5 (C, Density): no value"),
        (lambda r: r[4].__setitem__(0, "A"), "A5: material A appears twice"),
        (lambda r: r[4].__setitem__(0, None), "A5: no material name"),
    ]
    for change, says in cases:
        rows = [list(r) for r in GOOD]
        change(rows)
        errors, _ = errors_of(rows)
        assert any(says in e for e in errors), (says, errors)
    rows = [list(r) for r in GOOD]
    rows[1][2] = None                                        # Youngs_Modulus without units
    assert any("C2: no units for Youngs_Modulus" in e for e in errors_of(rows)[0])
    rows = [list(r) for r in GOOD]
    rows[1][1] = "MPa"                                       # not a unit of density
    assert any("B2: 'MPa' is not a unit of Density" in e for e in errors_of(rows)[0])
    errors, _ = errors_of(GOOD[:3])
    assert "at least two materials" in errors[0]
    errors, _ = errors_of([["Attributes", "Poissons_Ratio"], ["Material/Units", "-"], ["A", 0.3], ["B", 0.6]])
    assert "-1 < value < 0.5" in errors[0], errors
    assert "ATTRIBUTES in 0-Common/databases.py" in errors_of([["Attributes", "Colour"], ["u", "-"],
                                                               ["A", 1], ["B", 2]])[0][0]


def test_warnings_and_aliases():
    rows = [["Attributes", "youngs modulus", "Poisson_Ratio"], ["Material/Units", "Pa", "-"],
            ["A", 1.0, 0.3], ["B", 2.0, 0.3]]
    errors, warnings = errors_of(rows)
    assert errors == [] and len(warnings) == 1 and "Poissons_Ratio is 0.3 for every material" in warnings[0]
    assert attribute_name("Poisson_Ratio") == "Poissons_Ratio" and attribute_name("Colour") == "Colour"


def test_display():
    """The Display sheet: each material's colour and label, matched by name; empty means the
    default colour and the material's name. Two materials shown the same way are an error."""
    head = ["Material", "Color", "Label"]
    with tempfile.TemporaryDirectory() as d:
        assert material_display(write(os.path.join(d, "a.xlsx"), GOOD)) == (default_colours(3), ["A", "B", "C"])
        path = write(os.path.join(d, "b.xlsx"), GOOD, [head, ["C", "#00FF00", "steel"], ["A", None, "alu"],
                                                       ["B", "#123abc", None]])
        assert material_display(path) == ([default_colours(1)[0], "#123abc", "#00ff00"], ["alu", "B", "steel"])
    cases = [
        ([head, ["D", "#ff0000", None]], "sheet Display, A2: D is not a material of the first sheet"),
        ([head, ["A", "#ff0000", None], ["A", None, "x"]], "A3: A appears twice (also row 2)"),
        ([head, ["A", "red", None]], "B2 (A): 'red' is not a colour; write it as #rrggbb"),
        ([head, [None, "#ff0000", None]], "A2: no material name"),
        ([["Material", "Colour", "Name"]], "C1: 'Name'; row 1 holds Material, Color and Label"),
        ([head, ["A", "#FF0000", None], ["B", "#ff0000", None]], "A, B have the same colour, #ff0000"),
        ([head, ["B", default_colours(1)[0], None]], "A, B have the same colour"),                 # A's default
        ([head, ["A", None, "Steel 1"], ["B", None, "steel1"]], "A, B have the same label, steel1"),
        ([head, ["A", None, "b"]], "A, B have the same label, b"),                                # B's name
    ]
    for display, says in cases:
        errors, _ = errors_of(GOOD, display)
        assert any(says in e for e in errors), (says, errors)
    errors, warnings = errors_of(GOOD, [head, ["A", "#ff0000", None]])
    assert errors == [] and warnings == ["sheet Display has no row for B, C; these use the default colours"]

    many = [GOOD[0], GOOD[1]] + [[f"M{i}", 1.0 + i, 100.0 + i, 90.0 + i] for i in range(25)]
    errors, warnings = errors_of(many)
    assert errors == [] and len(warnings) == 1 and "25 materials and no colours given" in warnings[0], warnings
    assert len(set(default_colours(200))) == 200 and default_colours(25)[:20] == default_colours(20)
    with tempfile.TemporaryDirectory() as d:
        try:
            material_display(write(os.path.join(d, "c.xlsx"), GOOD, [head, ["A", "red", None]]))
        except ValueError as e:
            assert "'red' is not a colour" in str(e)
        else:
            raise AssertionError("a bad colour was accepted")


def test_shipped_display():
    """BM1 draws its materials in the colours their names say; the airplane bracket's alloys
    are labelled A to T, as in the paper."""
    bm1 = material_display(os.path.join(SHIPPED, "BM1_3Mats_2Attr.xlsx"))
    assert bm1 == (["#ff0000", "#4b00f1", "#23ff24"], ["Mat1_Red", "Mat2_Blue", "Mate3_Green"]), bm1
    colours, labels = material_display(os.path.join(SHIPPED, "CustomAlloys20_3D.xlsx"))
    assert colours == default_colours(20) and labels == [chr(ord("A") + i) for i in range(20)]


def test_template():
    with tempfile.TemporaryDirectory() as d:
        path = write_template(os.path.join(d, "t.xlsx"), ["Density", "Youngs_Modulus", "Yield"], n_materials=3)
        wb = load_workbook(path)
        assert wb.sheetnames == ["Main", "Display", "Reference"]
        disp = wb["Display"]
        assert [c.value for c in disp[2]] == ["Material1", default_colours(1)[0], None]
        assert disp["B2"].fill.fgColor.rgb.lower().endswith(default_colours(1)[0][1:])
        main = wb["Main"]
        assert [c.value for c in main[1]] == ["Attributes", "Density", "Youngs_Modulus", "Yield_Strength"]
        lists = [v for v in main.data_validations.dataValidation if v.type == "list"]
        assert any("MPa" in v.formula1 and "C2" in str(v.sqref) for v in lists)
        rules = {str(v.sqref): v.formula1 for v in main.data_validations.dataValidation if v.type == "custom"}
        assert rules["C3:C5"] == "AND(ISNUMBER(C3),C3>0)"         # E = 0 refused, as by the check
        assert wb["Reference"].max_row == 1 + len(ATTRIBUTES)
        errors, _ = check_database(path)                       # empty: every value is missing
        assert any("no value" in e for e in errors)
        for r, values in enumerate([(1.0, 100.0, 100.0), (1.2, 160.0, 140.0), (0.8, 70.0, 90.0)], start=3):
            for c, v in enumerate(values, start=2):
                main.cell(row=r, column=c, value=v)
        wb.save(path)
        assert check_database(path) == ([], [])
        try:
            write_template(os.path.join(d, "u.xlsx"), ["Colour"])
        except ValueError as e:
            assert "not known attributes: Colour" in str(e)
        else:
            raise AssertionError("an unknown attribute was accepted")


def test_units():
    assert unit_ratio("stress", "Pa", "mmKS") == 1e6 and unit_ratio("stress", "MPa", "mmKS") == 1.0
    assert abs(unit_ratio("density", "lb/in^3", "IPS") - 386.0886) < 1e-3     # pound-mass vs pound-force: g
    errors, _ = errors_of([["Attributes", "Youngs_Modulus", "Poissons_Ratio"], ["Material/Units", "Pa", None],
                           ["A", 1.0, 0.3], ["B", 2.0, 0.35]])
    assert errors == []                                      # a dimensionless column may have no units
    with tempfile.TemporaryDirectory() as d:
        beam = os.path.join(SHIPPED, "Beam_8Mats_3Attr_v4_spectrum2D.xlsx")
        out = os.path.join(d, "beam_mmKS.xlsx")
        changed = convert_database(beam, "mmKS", out)
        assert changed == ["Density: kg/m^3 -> t/mm^3", "Youngs_Modulus: Pa -> MPa"], changed
        assert check_database(out) == ([], [])
        assert database_units(out) == {"Density": "t/mm^3", "Youngs_Modulus": "MPa", "Yield_Strength": "MPa"}
        from databases import _read  # type: ignore
        a, b = _read(beam), _read(out)
        assert float(b.iat[2, 2]) == float(a.iat[2, 2]) / 1e6 and float(b.iat[2, 3]) == float(a.iat[2, 3])


def test_benchmark_unit_warnings():
    """The benchmarks declare the units they are set up in; the paper's databases mix units,
    and each mismatch is reported with what it does to the results."""
    from project import Project  # type: ignore
    counts = {}
    for name in ("L-bracket", "U-bracket", "beam", "Airplane bracket"):
        p = Project.load(os.path.join(ROOT, "projects", f"{name}.mmto"))
        counts[name] = (p.units, [w.split(" is in ")[0] for w in p.warnings()])
    assert counts == {"L-bracket": ("mmKS", ["Youngs_Modulus"]), "U-bracket": ("none", []),
                      "beam": ("mmKS", ["Density", "Youngs_Modulus"]),
                      "Airplane bracket": ("mmKS", ["Density", "Youngs_Modulus", "Cost"])}, counts
    w = Project.load(os.path.join(ROOT, "projects", "Airplane bracket.mmto")).warnings()
    assert "the bound 4e+07 means 4e-05 t" in w[0] and "the bound 7.5e+08 means 0.75 USD" in w[2]


def test_project_uses_the_check():
    from project import Project  # type: ignore
    ubracket = os.path.join(ROOT, "0-Common", "Models", "UBracket")
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(os.path.join(ubracket, "UBracket.STL"), d)
        p = Project("u", stl="UBracket.STL", base_dir=d)
        p.add_selection("bottom", __import__("selections").face("ymin"))
        p.add_support("bottom")
        p.add_force("bottom", total=[1, 0, 0])
        p.objective("FailureFactorPNorm()")
        p.constraint("Mass()", "<=", 1.0)

        p.database = "BM1_3Mats_2Attr"                         # no Yield_Strength, no Density
        errors = "\n".join(p.validate())
        assert "FailureFactorPNorm needs a Yield_Strength column" in errors, errors
        assert "Mass needs a Density column" in errors, errors

        bad = [list(r) for r in GOOD]
        bad[3][2] = -1.0
        p.database = write(os.path.join(d, "bad.xlsx"), bad)
        assert any("bad.xlsx: C4 (B, Youngs_Modulus)" in e for e in p.validate())

        p.database = write(os.path.join(d, "good.xlsx"), GOOD)
        assert p.validate() == []
        p.complete_settings()
        n_z = recommended_latent_dim(p.database)[1]
        assert p.settings["vae"]["latentDim"] == n_z                # a new database starts from the recommendation
        assert p.warnings() == []
        p.settings["vae"]["latentDim"] = n_z + 2
        assert any("is recommended" in w for w in p.warnings())
        p.settings["vae"]["latentDim"] = n_z

        p.units = "mmKS"                                       # GOOD is in Pa: Y too
        assert "the design is unaffected" in "\n".join(p.warnings())          # the failure factor as objective
        p.objective("Compliance()")
        p.constraint("FailureFactorPNorm()", "<=", 1.0)                       # and as a constraint
        assert "FailureFactorPNorm() <= 1 is a different problem" in "\n".join(p.warnings())
        p.units = "Imperial"
        assert any("units 'Imperial'" in e for e in p.validate())
        p.units = "none"
        assert p.units == "none" and p.warnings() == []


def test_vae_check():
    """A database's network is the .nt next to it with the same name. A trained network must fit the
    settings and the database; a run never trains into the shipped folder; replacing a network warns."""
    from project import Project, network_shape  # type: ignore
    for name in ("L-bracket", "U-bracket", "beam", "Airplane bracket", "rectangular beam"):
        p = Project.load(os.path.join(ROOT, "projects", f"{name}.mmto"))
        errors, warnings, status = p.vae_check()
        assert errors == [] and warnings == [] and status.endswith("shipped."), (name, errors, warnings)
    p = Project.load(os.path.join(ROOT, "projects", "L-bracket.mmto"))
    p.settings["vae"]["latentDim"] = 4
    assert p.vae_check()[0] == ["latentDim is 4, but BM1_3Mats_2Attr.nt was trained with 2: set it to 2, or "
                                "switch off use_pretrained_vae to train a new network"]
    assert any("latentDim is 4, but" in e for e in p.validate())
    p.settings["vae"]["use_pretrained_vae"] = False
    assert "save it over the shipped BM1_3Mats_2Attr.nt" in p.vae_check()[0][0]
    p.settings["vae"]["latentDim"] = 1                                       # smaller than recommended
    assert "smaller than the database needs cannot reproduce every material" in p.vae_check()[1][0]

    with tempfile.TemporaryDirectory() as d:
        bm1 = os.path.join(SHIPPED, "BM1_3Mats_2Attr")
        shutil.copy(bm1 + ".xlsx", os.path.join(d, "ggd.xlsx"))
        p = Project.load(os.path.join(ROOT, "projects", "L-bracket.mmto"))
        p.base_dir, p.database = d, "ggd.xlsx"
        assert p.network_path() == os.path.join(d, "ggd.nt")
        assert p.vae_check() == ([], [], "Network: ggd.nt, none yet. The first run trains ggd.nt.")
        shutil.copy(bm1 + ".nt", os.path.join(d, "ggd.nt"))
        assert p.vae_check() == ([], [], "Network: ggd.nt, found.") and network_shape(p.network_path()) == (2, 50, 2)
        p.settings["vae"]["use_pretrained_vae"] = False                          # retrain: allowed, with a warning
        errors, warnings, _ = p.vae_check()
        assert errors == [] and warnings == ["ggd.nt exists: the next run trains a new network and replaces it "
                                             "(the old one is kept as ggd.previous.nt). To keep both, copy "
                                             "ggd.xlsx first."]
        p.settings["vae"]["use_pretrained_vae"] = True
        shutil.copy(os.path.join(SHIPPED, "CustomAlloys20_3D.nt"), os.path.join(d, "ggd.nt"))   # another database's
        errors = p.vae_check()[0]
        assert "ggd.nt takes 4 attributes, but ggd.xlsx has 2: it was trained on another database" in errors
        open(os.path.join(d, "ggd.nt"), "w").write("not a network")
        assert p.vae_check()[0] == ["ggd.nt could not be read as a VAE network"]


if __name__ == "__main__":
    failed = 0
    for name, fn in [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]:
        try:
            fn()
            print(f"ok   {name}")
        except Exception as e:                                     # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)
