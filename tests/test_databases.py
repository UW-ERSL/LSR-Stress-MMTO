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

from openpyxl import Workbook, load_workbook  # noqa: E402
from lsr_mmto.materials.databases import (ATTRIBUTES, DATABASE_DIR, DATABASE_VAE, VAEParams,  # noqa: E402
                                         attribute_name, check_database, database_units, default_colours,
                                         latent_dim_breakdown, material_display, recommended_latent_dim,
                                         unit_ratio, write_template)

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
    from lsr_mmto.materials.encoder import MaterialEncoder
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


def test_decode_undoes_normalize_by_the_attributes_flag(monkeypatch):
    """The network's output is turned back into values by the same flag normalize scales by: a column whose
    attribute is log_scaled comes back through 10 **, a linear one directly; each round trip gives the
    database's values to 1e-10. Nothing is named by hand: a column named E0 is linear like any other."""
    import numpy as np
    from lsr_mmto.materials.encoder import MaterialEncoder
    monkeypatch.setattr(ATTRIBUTES["Youngs_Modulus"], "log_scaled", True)
    with tempfile.TemporaryDirectory() as d:
        rows = [r[:] for r in GOOD]
        rows[2][2], rows[3][2], rows[4][2] = 2e9, 2e11, 7e10              # three orders of magnitude
        for r, v in zip(rows, ["E0", "-", 1.0, 10.0, 100.0]):             # a name once decoded by hand
            r.append(v)
        enc = MaterialEncoder(VAEParams())
        enc.readExcel(write(os.path.join(d, "db.xlsx"), rows))
    props = enc.getMaterialProperties(enc.scaledMaterialData)            # normalize, then decode
    for name, a in enc.materialAttributes.items():
        raw = np.asarray(enc.rawData, dtype=float)[:, a["idx"]]
        assert np.allclose(props[name].numpy(), raw, rtol=1e-10, atol=0), (name, props[name], raw)
    E = enc.scaledMaterialData[:, enc.materialAttributes["Youngs_Modulus"]["idx"]].numpy()
    logs = np.log10([2e9, 2e11, 7e10])
    assert np.allclose(E, (logs - logs.min()) / (logs.max() - logs.min()), rtol=0, atol=1e-9)   # log-scaled in


def test_errors_name_the_cell():
    assert errors_of(GOOD) == ([], [])
    cases = [
        (lambda r: r[0].__setitem__(2, "Youngs_Modulous"), "C1: 'Youngs_Modulous' is not a property LSR MMTO knows"),
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
    rows = [list(r) for r in GOOD]
    rows[4][1:] = rows[2][1:]                                # C has A's numbers
    assert errors_of(rows)[0] == ["A3, A5: A, C have the same value in every attribute, so the VAE cannot "
                                  "tell them apart; keep one, or correct a value"]
    errors, _ = errors_of([["My materials"]] + [list(r) for r in GOOD])     # a title row above the table
    assert errors == ["row 1 is mostly empty and row 2 holds attribute names (Density, Youngs_Modulus, "
                      "Yield_Strength): the table seems to start a row down. Row 1 must hold the attribute "
                      "names, row 2 the units, and the materials start in row 3"], errors
    errors, _ = errors_of([["Attributes", "Poissons_Ratio"], ["Material/Units", "-"], ["A", 0.3], ["B", 0.6]])
    assert "-1 < value < 0.5" in errors[0], errors
    unknown = errors_of([["Attributes", "Colour"], ["u", "-"], ["A", 1], ["B", 2]])[0][0]
    assert unknown.startswith("B1: 'Colour' is not a property LSR MMTO knows (Density, ") and \
        unknown.endswith("Rename the column to one of them, or use Import..., which leaves out the columns it does "
                         "not know.") and "lsr_mmto" not in unknown, unknown


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
        assert [c.value for c in disp[2]] == ["=Main!A3", default_colours(1)[0], None]     # follows Main
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
        for r, name in enumerate(["Soft", "Mid", "Stiff"], start=3):        # renamed on Main only: the
            main.cell(row=r, column=1, value=name)                          # Display sheet follows (a file
        wb.save(path)                                                       # saved with no formula values)
        assert check_database(path) == ([], [])
        assert material_display(path) == (default_colours(3), ["Soft", "Mid", "Stiff"])
        main.delete_rows(5)                                                 # a material removed on Main:
        wb.save(path)                                                       # its Display row is skipped
        assert check_database(path) == ([], []) and material_display(path)[1] == ["Soft", "Mid"]
        wb["Display"]["A4"] = "=Main!#REF!"                                 # the same, deleted in Excel
        wb.save(path)
        assert check_database(path) == ([], []) and material_display(path)[1] == ["Soft", "Mid"]
        d_eff, n_z, ratios = latent_dim_breakdown(path)
        assert (d_eff, n_z) == recommended_latent_dim(path) and abs(sum(ratios) - 1.0) < 1e-12
        assert list(ratios) == sorted(ratios, reverse=True)
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


def test_benchmark_unit_warnings():
    """The benchmarks declare the units they are set up in; the paper's databases mix units,
    and each mismatch is reported with what it does to the results."""
    from lsr_mmto.problem.project import Project
    counts = {}
    for name in ("L-bracket", "U-bracket", "beam", "Airplane bracket"):
        p = Project.load(os.path.join(ROOT, "projects", f"{name}.mmto"))
        counts[name] = (p.units, [w.split("the database gives ")[1].split(" in ")[0] for w in p.warnings()])
    assert counts == {"L-bracket": ("mmKS", ["Young's modulus"]), "U-bracket": ("none", []),
                      "beam": ("mmKS", ["Density", "Young's modulus"]),
                      "Airplane bracket": ("mmKS", ["Density", "Young's modulus", "Cost"])}, counts
    w = Project.load(os.path.join(ROOT, "projects", "Airplane bracket.mmto")).warnings()
    assert "limit of 4e+07 is really a limit of 4e-05 t" in w[0] and "limit of 7.5e+08 is really a limit of 0.75 USD" in w[2]


def test_a_mass_limit_far_below_the_lightest_solid_part_is_warned():
    """A Mass limit below 1 % of the part solid in the lightest material is a warning (not an error), with
    the share it allows to 2 significant digits: the U-bracket (volume 0.00875) with Beam_8Mats (lightest
    density 1500) and Mass <= 0.0035 kg allows 0.0035 / (1500 * 0.00875) = 0.027 %."""
    from lsr_mmto.problem.project import Project
    p = Project.load(os.path.join(ROOT, "projects", "U-bracket.mmto"))
    p.database, p.units = "Beam_8Mats_3Attr_v4_spectrum2D", "MKS"
    p.formulation["constraints"] = []
    p.constraint("Mass()", "<=", 0.0035)
    assert f"{100 * 0.0035 / (1500 * 0.00875):.2g}" == "0.027"
    said = "constraint 1 (Mass <= 0.0035): This limit allows at most 0.027% of the part even in the lightest material."
    assert said in p.warnings() and not any("lightest" in e for e in p.validate())
    p.formulation["constraints"][0]["bound"] = 0.01 * 1500 * 0.00875 * 1.001     # just above 1 %: no warning
    assert not any("lightest material" in w for w in p.warnings())
    p.formulation["constraints"][0]["bound"] = 0.01 * 1500 * 0.00875 * 0.999
    assert any("at most 1% of the part" in w for w in p.warnings())


def test_project_uses_the_check():
    from lsr_mmto.problem.project import Project
    ubracket = os.path.join(ROOT, "data", "models", "UBracket")
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(os.path.join(ubracket, "UBracket.STL"), d)
        p = Project("u", stl="UBracket.STL", base_dir=d)
        p.add_selection("bottom", __import__("lsr_mmto").face("ymin"))
        p.add_bc("fixed", "bottom", dofs="xyz")
        p.add_bc("force", "bottom", total=[1, 0, 0])
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
        assert "not the design" in "\n".join(p.warnings())          # the failure factor as objective
        p.objective("Compliance()")
        p.constraint("FailureFactorPNorm()", "<=", 1.0)                       # and as a constraint
        assert "Failure factor (p-norm) comes out" in "\n".join(p.warnings())
        p.units = "Imperial"
        assert any("units 'Imperial'" in e for e in p.validate())
        p.units = "none"
        assert p.units == "none" and p.warnings() == []


def test_vae_check():
    """A database's network is the .nt next to it with the same name. A trained network must fit the
    settings and the database; a run never trains into the shipped folder; replacing a network warns."""
    from lsr_mmto.problem.project import Project, network_shape
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



def test_a_trained_network_sets_its_own_shape(capsys):
    """A database chosen with a trained network of another shape than the settings say: the settings that
    give the network its shape follow the network, the change is said, and validate finds nothing wrong.
    A later explicit change is still checked; a project read from its file keeps its settings as saved."""
    from lsr_mmto.problem.project import Project
    with tempfile.TemporaryDirectory() as d:
        bm1 = os.path.join(SHIPPED, "BM1_3Mats_2Attr")
        shutil.copy(bm1 + ".xlsx", os.path.join(d, "ggd.xlsx"))
        shutil.copy(bm1 + ".nt", os.path.join(d, "ggd.nt"))                  # hidden width 50, latent 2
        p = Project.load(os.path.join(ROOT, "projects", "L-bracket.mmto"))
        p.stl, p.base_dir = p.stl_path(), d                                   # the database in d, the part kept
        p.settings["vae"]["vae_hiddenDim"] = 300
        p.database = "ggd.xlsx"
        assert p.settings["vae"]["vae_hiddenDim"] == 50 and p.settings["vae"]["latentDim"] == 2
        assert "Network settings set to match the trained network: hidden layer width 50" in capsys.readouterr().out
        assert p.validate() == []
        assert p.match_network() is None                                      # nothing left to change
        p.settings["vae"]["vae_hiddenDim"] = 300                              # a later change: checked as before
        assert any("vae_hiddenDim is 300, but ggd.nt was trained with 50" in e for e in p.validate())
        path = os.path.join(d, "P.mmto")
        p.save(path)
        assert Project.load(path).settings["vae"]["vae_hiddenDim"] == 300     # read back as saved
        q = Project.load(os.path.join(ROOT, "projects", "L-bracket.mmto"))
        q.base_dir = d
        q.settings["vae"]["use_pretrained_vae"] = False                       # training a new one: any shape
        q.settings["vae"]["vae_hiddenDim"] = 300
        q.database = "ggd.xlsx"
        assert q.settings["vae"]["vae_hiddenDim"] == 300


def test_a_hand_set_network_is_cleared_on_another_database(capsys):
    """A VAE file set by hand (settings.vae.saveNet) belongs to the database it was set for. The checks refuse a
    network of another number of columns, but not one trained on another database of as many columns, so
    choosing another database clears it, and says so; the network next to the new database is then used.
    Setting the same database again keeps it."""
    from lsr_mmto.problem.project import Project
    with tempfile.TemporaryDirectory() as d:
        bm1 = os.path.join(SHIPPED, "BM1_3Mats_2Attr")
        shutil.copy(bm1 + ".xlsx", os.path.join(d, "mine.xlsx"))
        shutil.copy(bm1 + ".nt", os.path.join(d, "other.nt"))                 # trained on another database
        p = Project.load(os.path.join(ROOT, "projects", "L-bracket.mmto"))
        p.stl, p.base_dir = p.stl_path(), d
        p.settings["vae"]["saveNet"] = os.path.join(d, "other.nt")
        p.database = p.database                                               # the same one: kept
        assert p.settings["vae"]["saveNet"] == os.path.join(d, "other.nt")
        capsys.readouterr()
        p.database = "mine.xlsx"
        assert "saveNet" not in p.settings["vae"] and p.network_path() == os.path.join(d, "mine.nt")
        assert capsys.readouterr().out.startswith("[L-bracket] The VAE file set by hand (other.nt) was for the "
                                                  "database chosen before, so it no longer applies: the network "
                                                  "next to the database is used.")


def test_read_once_per_file_version():
    """A database is read once for each version of its file: looked at again it is not read again, a
    copy is handed back each time, and once the file is saved again it is read again."""
    import time
    from lsr_mmto.materials import databases
    calls = []

    @databases.once_per_file_version
    def rows(path):
        calls.append(path)
        return [list(r) for r in load_workbook(path).active.iter_rows(values_only=True)]

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "db.xlsx")
        write(path, GOOD)
        first = rows(path)
        first[2][1] = "changed by the caller"
        assert rows(path)[2][1] == 1.0 and len(calls) == 1          # kept, and a copy each time
        time.sleep(0.01)
        write(path, GOOD[:-1])                                       # saved again: read again
        assert len(rows(path)) == 4 and len(calls) == 2
        assert check_database(path) == check_database(path)



def test_import_table():
    """A table made elsewhere: the first column the names, row 1 the properties under any of their names
    (with a unit in brackets or not), an optional units row as in a database of LSR MMTO; columns that are
    not a known property, a second column of the same property and units a property does not take are
    reported, not read."""
    from lsr_mmto.materials.databases import import_table
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.xlsx")
        write(path, [["Material", "Young's modulus", "nu", "Hardness", "E", "Density"],
                     ["Units", "GPa", "", "HV", "GPa", "furlongs"],
                     ["A", 70, 0.3, 100, 1, 2700], ["B", 110, 0.34, 300, 2, 4400], [None, None, None, None, None, None]])
        t = import_table(path)
        assert t["names"] == ["A", "B"] and t["attributes"] == ["Youngs_Modulus", "Poissons_Ratio", "Density"]
        assert t["units"] == ["GPa", None, None] and t["rows"] == [["70", "0.3", "2700"], ["110", "0.34", "4400"]]
        assert t["skipped"] == ["Hardness", "E"] and t["unknown_units"] == [("Density", "furlongs")]
        t = import_table(os.path.join(SHIPPED, "BM1_3Mats_2Attr.xlsx"))             # a database of LSR MMTO
        assert t["attributes"] == ["Youngs_Modulus", "Poissons_Ratio"] and not t["skipped"] and len(t["names"]) == 3
        write(path, [["Material", "Hardness"], ["A", 1]])
        try:
            import_table(path)
            raise AssertionError("a table with no known property was read")
        except ValueError as e:
            assert "none of the columns of t.xlsx is a property LSR MMTO knows" in str(e)


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


def test_unit_kinds_of_a_physics():
    """A physics' own kinds of quantity (UNIT_KINDS on its front page) are in the unit tables; a kind
    given differently by two physics, or a unit system's unit that is not one of the kind's units, is
    refused."""
    import types
    import pytest
    from lsr_mmto.materials import databases as D
    assert D.UNITS["stress"]["MPa"] == 1e6 and D.UNIT_SYSTEMS["IPS"]["force"] == "lbf"
    other = types.SimpleNamespace(NAME="other", UNIT_KINDS={"force": {"units": {"N": 1.0}, "MKS": "N",
                                                                       "mmKS": "N", "IPS": "N"}})
    with pytest.raises(ValueError, match="differ"):
        D._add_unit_kinds(other)
    odd = types.SimpleNamespace(NAME="odd", UNIT_KINDS={"heat": {"units": {"W": 1.0}, "MKS": "W", "mmKS": "mW",
                                                                  "IPS": "W"}})
    with pytest.raises(ValueError, match="not one of its units"):
        D._add_unit_kinds(odd)
    assert "heat" not in D.UNITS                                    # nothing of a refused kind is kept


def test_columns_found_by_what_they_measure():
    """The mass of a design is made from the column that measures density, and the per-iteration
    materials line prints the columns whose attribute has a label there, in their order, whatever
    the columns are named."""
    from lsr_mmto.materials.databases import density_column, iteration_columns
    assert density_column(["Youngs_Modulus", "Density"]) == "Density"
    assert density_column(["Youngs_Modulus"]) is None
    assert iteration_columns(["Density", "Youngs_Modulus", "Yield_Strength"]) == [("Yield_Strength", "Y"),
                                                                                   ("Density", "rho")]


def test_a_blank_unit_is_no_unit_for_the_network_too():
    """A blank unit cell (BM1_3Mats_2Attr's Poisson's ratio, a ratio) is no unit, None, for the material network
    as for the database's own reader, so the network's error table shows nothing there, not "nan"."""
    from lsr_mmto.materials.databases import database_table
    from lsr_mmto.materials.encoder import MaterialEncoder
    from lsr_mmto.results.tables import encoding_errors
    path = os.path.join(DATABASE_DIR, "BM1_3Mats_2Attr.xlsx")
    params = VAEParams()
    params.latentDim, params.vae_hiddenDim = DATABASE_VAE["BM1_3Mats_2Attr"]["latentDim"], \
        DATABASE_VAE["BM1_3Mats_2Attr"]["vae_hiddenDim"]
    enc = MaterialEncoder(params)
    enc.readExcel(path)
    assert database_table(path)[2] == ["Pa", None]
    assert {k: a["unit"] for k, a in enc.materialAttributes.items()} == {"Youngs_Modulus": "Pa", "Poissons_Ratio": None}
    enc.loadAutoencoderFromFile(os.path.splitext(path)[0] + ".nt")
    assert "nan" not in repr(encoding_errors(enc)) and "nan" not in repr(encoding_errors(enc, per_material=True))

