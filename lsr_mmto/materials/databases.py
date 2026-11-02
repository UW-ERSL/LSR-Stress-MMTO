"""
databases.py
============
The material databases shipped with the repository, and the VAE settings each was
trained with, so the network shipped beside each database (.nt) can be loaded.

A database is an .xlsx file in DATABASE_DIR: row 1 the attribute names, row 2 their
units, then one row per material. An optional sheet "Display" gives each material the
colour and the label it is shown with (material_display).
"""
import os
from dataclasses import dataclass

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))     # the repository folder
DATABASE_DIR = os.path.join(REPO, "data", "databases")


def in_shipped_folder(path):
    """True when path is in DATABASE_DIR, the folder of the shipped databases and their networks,
    which a run never writes into."""
    norm = lambda p: os.path.normcase(os.path.realpath(p))                   # noqa: E731
    return norm(os.path.dirname(os.path.abspath(path))) == norm(DATABASE_DIR)


@dataclass(slots=True)
class VAEParams:
    """VAE settings for a database that has none of its own."""
    klFactor: float = 5e-6
    learningRate: float = 2e-6
    numEpochs: int = 200000                 # max epochs
    vae_hiddenDim: int = 500
    latentDim: int = 2
    maxAttributeErrorPercent: float = 1     # stops training when every attribute is this close


# the settings each shipped network was trained with
DATABASE_VAE = {
    "BM1_3Mats_2Attr":                dict(latentDim=2, learningRate=5e-4, vae_hiddenDim=50,
                                           maxAttributeErrorPercent=0.1, klFactor=1e-4, numEpochs=30000),
    "6Materials_AF_3Attr":            dict(latentDim=2, learningRate=1e-4, vae_hiddenDim=50,
                                           maxAttributeErrorPercent=0.1, klFactor=1e-4, numEpochs=100000),
    "Beam_8Mats_3Attr_v4_spectrum2D": dict(latentDim=2, learningRate=5e-4, vae_hiddenDim=50,
                                           maxAttributeErrorPercent=0.1, klFactor=1e-4, numEpochs=100000),
    "CustomAlloys20_3D":              dict(latentDim=3, learningRate=5e-4, vae_hiddenDim=50,
                                           maxAttributeErrorPercent=0.1, klFactor=1e-4, numEpochs=100000),
}


# ---------------------------------------------------------------------- units
# Nothing is converted: numbers are used as they are written. Units are declared (row 2 of
# a database, the project's unit system) so that a mismatch can be reported.
_IN, _LBF, _LB = 0.0254, 4.4482216152605, 0.45359237           # inch in m, pound-force in N, pound in kg
_SLINCH = _LBF / _IN                                            # lbf s^2/in, the IPS unit of mass, in kg

# what one unit is in SI, for each kind of quantity
UNITS = {
    "length": {"m": 1.0, "mm": 1e-3, "in": _IN},
    "force": {"N": 1.0, "lbf": _LBF},
    "stress": {"Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9, "psi": _LBF / _IN ** 2, "ksi": 1e3 * _LBF / _IN ** 2},
    "density": {"kg/m^3": 1.0, "g/cm^3": 1e3, "t/mm^3": 1e12, "lb/in^3": _LB / _IN ** 3,
                "lbf*s^2/in^4": _SLINCH / _IN ** 3},
    "mass": {"kg": 1.0, "t": 1e3, "lbf*s^2/in": _SLINCH},
    "cost per mass": {"USD/kg": 1.0, "USD/g": 1e3, "USD/t": 1e-3, "USD/lb": 1 / _LB, "USD/(lbf*s^2/in)": 1 / _SLINCH},
    "ratio": {"-": 1.0},
}

# The unit systems of the PyTO-AD GUI (settings.unit_system in a .mmto file). A project
# without one is "none": its units are not checked.
UNIT_SYSTEMS = {
    "MKS":  {"length": "m", "force": "N", "stress": "Pa", "density": "kg/m^3", "mass": "kg",
             "cost per mass": "USD/kg"},
    "mmKS": {"length": "mm", "force": "N", "stress": "MPa", "density": "t/mm^3", "mass": "t",
             "cost per mass": "USD/t"},
    "IPS":  {"length": "in", "force": "lbf", "stress": "psi", "density": "lbf*s^2/in^4", "mass": "lbf*s^2/in",
             "cost per mass": "USD/(lbf*s^2/in)"},
}


def unit_ratio(measures, unit, system):
    """How many times larger a number written in unit is than the same value in the unit
    system's unit: 10^6 for a stress in Pa in an mmKS project."""
    return UNITS[measures][UNIT_SYSTEMS[system][measures]] / UNITS[measures][unit]


# ---------------------------------------------------------------------- attributes
class Attribute:
    """One material attribute a database may hold. measures: what it is, one of UNITS
    (stress, density, ratio, ...); its units are those of UNITS[measures], and "ratio"
    means dimensionless (its units cell may be left blank). low, high: the allowed range,
    each end open unless named in closed."""

    def __init__(self, name, measures, doc, low=None, high=None, closed=(), aliases=()):
        if measures not in UNITS:
            raise ValueError(f"attribute {name}: measures must be one of {', '.join(UNITS)}")
        self.name, self.measures, self.doc = name, measures, doc
        self.low, self.high, self.closed = low, high, tuple(closed)
        self.units, self.aliases = tuple(UNITS[measures]), tuple(aliases)

    def in_range(self, v):
        if self.low is not None and (v < self.low or (v == self.low and "low" not in self.closed)):
            return False
        if self.high is not None and (v > self.high or (v == self.high and "high" not in self.closed)):
            return False
        return True

    def range_text(self):
        lo = "" if self.low is None else f"{self.low} {'<=' if 'low' in self.closed else '<'} "
        hi = "" if self.high is None else f" {'<=' if 'high' in self.closed else '<'} {self.high}"
        return f"{lo}value{hi}" if (lo or hi) else "any"


# Every attribute a database may hold. To add one, add an entry; the check and the Excel
# template follow. A column not listed here is an error.
ATTRIBUTES = {a.name: a for a in [
    Attribute("Density", "density", "mass per unit volume", low=0),
    Attribute("Youngs_Modulus", "stress", "Young's modulus, the stiffness", low=0,
              aliases=("Youngs Modulus", "Young_Modulus", "E")),
    Attribute("Poissons_Ratio", "ratio", "Poisson's ratio", low=-1, high=0.5,
              aliases=("Poisson_Ratio", "Poissons Ratio", "nu")),
    Attribute("Yield_Strength", "stress", "yield strength, the stress at which the material yields",
              low=0, aliases=("Yield Strength", "Yield")),
    Attribute("Cost", "cost per mass", "price per unit mass", low=0, closed=("low",)),
]}
# each attribute's name as a user reads it
SHOWN = {"Density": "Density", "Youngs_Modulus": "Young's modulus", "Poissons_Ratio": "Poisson's ratio",
         "Yield_Strength": "Yield strength", "Cost": "Cost"}
_BY_ALIAS = {k.lower(): a.name for a in ATTRIBUTES.values() for k in (a.name,) + a.aliases}


def attribute_name(name):
    """The registry name of a database column; the name itself when it is not in the registry."""
    return _BY_ALIAS.get(str(name).strip().lower(), name)


# ---------------------------------------------------------------------- reading and checking
def _read(path):
    import warnings
    import pandas as pd
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Data Validation extension")
        return pd.read_excel(path, header=None)


def normalize(values, attribute_names):
    """Each column scaled to [0, 1] (log10 first for the columns named E0..E3, Y0..Y3):
    the normalized data, and each column's minimum and maximum before scaling."""
    import numpy as np
    normalized = np.zeros_like(values)
    scale_min = np.zeros(values.shape[1])
    scale_max = np.zeros(values.shape[1])
    for i, name in enumerate(attribute_names):
        col = values[:, i]
        if name in ['E0', 'E1', 'E2', 'E3', 'Y0', 'Y1', 'Y2', 'Y3']:
            col = np.log10(col)
        scale_min[i] = col.min()
        scale_max[i] = col.max()
        normalized[:, i] = (col - scale_min[i]) / (scale_max[i] - scale_min[i] + 1e-12)
    return normalized, scale_min, scale_max


def participation_ratio(X):
    """d_eff = (sum of PCA eigenvalues)^2 / sum of their squares, with the eigenvalues
    and the explained variance ratios. 0 when every eigenvalue is zero."""
    import numpy as np
    from sklearn.decomposition import PCA
    pca = PCA().fit(X)
    lambdas = np.asarray(pca.explained_variance_, dtype=float)
    if np.all(lambdas <= 1e-16):
        return 0.0, lambdas, np.zeros_like(lambdas)
    return (float((np.sum(lambdas) ** 2) / np.sum(lambdas ** 2)), lambdas,
            np.asarray(pca.explained_variance_ratio_, dtype=float))


def database_attributes(path):
    """The registry names of a database's columns, in order."""
    return [attribute_name(n) for n in _read(path).iloc[0, 1:].tolist()]


def database_units(path):
    """Each column's unit as written in row 2 (None when blank), by registry name."""
    df = _read(path)
    out = {}
    for c in range(1, df.shape[1]):
        u = df.iat[1, c]
        out[attribute_name(df.iat[0, c])] = None if (u is None or str(u).strip() in ("", "nan")) else str(u).strip()
    return out


def database_table(path):
    """(names, attributes, units, values): the materials, the columns' registry names, each
    column's unit as written (None when blank) and the numbers, one row per material."""
    import numpy as np
    df = _read(path)
    names = [str(m).strip() for m in df.iloc[2:, 0].tolist()]
    attributes = [attribute_name(n) for n in df.iloc[0, 1:].tolist()]
    units = [None if (u is None or str(u).strip() in ("", "nan")) else str(u).strip() for u in df.iloc[1, 1:]]
    values = np.array([[float(v) for v in row] for row in df.iloc[2:, 1:].to_numpy()])
    return names, attributes, units, values


def recommended_latent_dim(path):
    """(d_eff, n_z): the participation ratio of the normalized database, and the latent
    dimension recommended for it, the smallest integer at or above d_eff (at least 1)."""
    d_eff, n_z, _ = latent_dim_breakdown(path)
    return d_eff, n_z


def latent_dim_breakdown(path):
    """(d_eff, n_z, ratios): as recommended_latent_dim, with the share of the variation each
    principal component of the normalized database explains (largest first, summing to 1)."""
    import math
    df = _read(path)
    names = [attribute_name(n) for n in df.iloc[0, 1:].tolist()]
    values = df.iloc[2:, 1:].to_numpy(dtype=float)
    d_eff, _, ratios = participation_ratio(normalize(values, names)[0])
    return d_eff, max(1, math.ceil(d_eff)), ratios


def check_database(path):
    """(errors, warnings) for a database file. A unit must be one of its attribute's; whether
    it suits a project's unit system is checked by the project (Project.warnings)."""
    import math
    import numpy as np
    errors, warnings_ = [], []
    if not os.path.isfile(path):
        return [f"{path} does not exist"], []
    try:
        df = _read(path)
    except Exception as e:                                          # noqa: BLE001
        return [f"{path} could not be read as an Excel file: {e}"], []

    cell = _cell

    def blank(v):
        return v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == ""

    if df.shape[1] < 2 or df.shape[0] < 4:
        return [f"the first sheet needs attribute names in row 1, units in row 2 and at least two "
                f"materials below, with the material names in column A; it has {df.shape[0]} rows "
                f"and {df.shape[1]} columns"], []

    blank_row1 = [c for c in range(1, df.shape[1]) if blank(df.iat[0, c])]
    known_row2 = [attribute_name(df.iat[1, c]) for c in range(1, df.shape[1])
                  if not blank(df.iat[1, c]) and attribute_name(df.iat[1, c]) in ATTRIBUTES]
    if 2 * len(blank_row1) > df.shape[1] - 1 and known_row2:
        return [f"row 1 is mostly empty and row 2 holds attribute names ({', '.join(known_row2)}): the table seems "
                f"to start a row down. Row 1 must hold the attribute names, row 2 the units, and the materials "
                f"start in row 3"], []

    names = {}
    for c in range(1, df.shape[1]):
        raw = df.iat[0, c]
        if blank(raw):
            errors.append(f"{cell(0, c)}: no attribute name")
            continue
        name = attribute_name(raw)
        if name not in ATTRIBUTES:
            errors.append(f"{cell(0, c)}: {raw!r} is not a known attribute ({', '.join(ATTRIBUTES)}); "
                          f"to use a new one, add it to ATTRIBUTES in lsr_mmto/materials/databases.py")
            continue
        if name in names.values():
            errors.append(f"{cell(0, c)}: {name} appears twice")
            continue
        names[c] = name
        a, u = ATTRIBUTES[name], df.iat[1, c]
        if blank(u):
            if a.measures != "ratio":
                errors.append(f"{cell(1, c)}: no units for {name}; one of {', '.join(a.units)}")
        elif str(u).strip() not in a.units:
            errors.append(f"{cell(1, c)}: {str(u).strip()!r} is not a unit of {name}; one of {', '.join(a.units)}")

    materials = {}
    for r in range(2, df.shape[0]):
        m = df.iat[r, 0]
        if blank(m):
            errors.append(f"{cell(r, 0)}: no material name")
        elif str(m) in materials.values():
            errors.append(f"{cell(r, 0)}: material {m} appears twice")
        else:
            materials[r] = str(m)

    for c, name in names.items():
        a, column = ATTRIBUTES[name], []
        for r in range(2, df.shape[0]):
            v = df.iat[r, c]
            where = f"{cell(r, c)} ({materials.get(r, 'row ' + str(r + 1))}, {name})"
            try:
                x = float(v)
            except (TypeError, ValueError):
                errors.append(f"{where}: {v!r} is not a number")
                continue
            if not np.isfinite(x):
                errors.append(f"{where}: no value" if blank(v) else f"{where}: {v!r} is not a finite number")
                continue
            if not a.in_range(x):
                errors.append(f"{where}: {x:g} is outside the allowed range, {a.range_text()}")
            column.append(x)
        if len(column) == df.shape[0] - 2 and len(set(column)) == 1:
            warnings_.append(f"{name} is {column[0]:g} for every material, so it tells the VAE nothing "
                             f"about the materials")

    if not errors and names:                                   # the VAE cannot tell identical materials apart
        same = {}
        for r in materials:
            same.setdefault(tuple(float(df.iat[r, c]) for c in names), []).append(r)
        for rows in same.values():
            if len(rows) > 1:
                errors.append(f"{', '.join(cell(r, 0) for r in rows)}: {', '.join(materials[r] for r in rows)} have "
                              f"the same value in every attribute, so the VAE cannot tell them apart; keep one, "
                              f"or correct a value")

    if not errors:
        _, _, display_errors, display_warnings = _display(path, [str(m).strip() for m in materials.values()])
        errors += display_errors
        warnings_ += display_warnings
    return errors, warnings_


# ---------------------------------------------------------------------- the Display sheet
# An optional sheet "Display": Material | Color | Label, one row per material, matched by
# name. Color is #rrggbb; empty means the default colour. Label empty means the material's
# name. Figures, tables and the run record take both from here.
DISPLAY_SHEET = "Display"
N_DEFAULT_COLOURS = 20


def default_colours(n):
    """The colour of each material with none given: materialColors.material_colors for the
    first 20, then colours generated by turning the hue by the golden angle and alternating
    the lightness. The same n gives the same colours; no two are the same."""
    import colorsys
    from lsr_mmto.materials.colors import material_colors
    out = [c.lower() for c in material_colors[:min(n, N_DEFAULT_COLOURS)]]
    taken, k = set(out), 0
    while len(out) < n:
        h = (k * 137.50776405) % 360 / 360
        light = (0.38, 0.55, 0.70)[k % 3]
        r, g, b = colorsys.hls_to_rgb(h, light, 0.75)
        c = "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))
        k += 1
        if c not in taken:
            out.append(c)
            taken.add(c)
    return out


def _linked_names(path, sheet, column):
    """{row index, as pandas reads the sheet: material name} for the Material cells that are a formula
    pointing at one cell of a sheet, as the template's =Main!A3. The name is read from that cell, so
    the Display sheet follows the names on the first sheet whatever program saved the file, also one
    that stores no value for a formula. A pointer to an empty cell gives ""."""
    import re
    import warnings
    from openpyxl import load_workbook
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Data Validation extension")
        wb = load_workbook(path, data_only=False)
    out = {}
    pointer = re.compile(r"=\s*(?:'([^']+)'|([^'!]+))!\$?([A-Za-z]{1,3})\$?(\d+)\s*")
    for (cell,) in wb[sheet].iter_rows(min_row=2, min_col=column + 1, max_col=column + 1):
        if isinstance(cell.value, str) and cell.value.startswith("=") and "#REF!" in cell.value.upper():
            out[cell.row - 1] = ""                                  # Excel, after its row on Main was deleted
            continue
        m = pointer.fullmatch(cell.value) if isinstance(cell.value, str) else None
        target = None if m is None else (m.group(1) or m.group(2)).strip()
        if target in wb.sheetnames:
            value = wb[target][f"{m.group(3).upper()}{m.group(4)}"].value
            out[cell.row - 1] = "" if value is None else str(value).strip()
    return out


def _display(path, names):
    """(colours, labels, errors, warnings) for the materials named, in that order."""
    import math
    import re
    import warnings
    import pandas as pd
    colours, labels = [None] * len(names), [None] * len(names)
    errors, warnings_ = [], []

    def blank(v):
        return v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == ""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Data Validation extension")
        with pd.ExcelFile(path) as xl:
            sheet = next((s for s in xl.sheet_names if s.strip().lower() == DISPLAY_SHEET.lower()), None)
            df = None if sheet is None else pd.read_excel(xl, sheet_name=sheet, header=None)

    if df is not None and df.shape[0] > 0:
        where = f"sheet {sheet}"
        heads = {}
        for c in range(df.shape[1]):
            h = df.iat[0, c]
            if blank(h):
                continue
            key = {"material": "material", "color": "color", "colour": "color", "label": "label"}.get(
                str(h).strip().lower())
            if key is None or key in heads:
                errors.append(f"{where}, {_cell(0, c)}: {h!r}; row 1 holds Material, Color and Label")
            else:
                heads[key] = c
        if "material" not in heads:
            errors.append(f"{where}: no Material column; row 1 holds Material, Color and Label")
        else:
            index = {n: i for i, n in enumerate(names)}
            seen = {}
            linked = _linked_names(path, sheet, heads["material"])
            for r in range(1, df.shape[0]):
                row = {k: df.iat[r, c] for k, c in heads.items()}
                if r in linked:                                     # =Main!A3: the name there, now
                    if linked[r] == "":                             # an empty cell: that material is gone
                        continue
                    row["material"] = linked[r]
                if all(blank(v) for v in row.values()):
                    continue
                m = row["material"]
                if blank(m):
                    errors.append(f"{where}, {_cell(r, heads['material'])}: no material name")
                    continue
                m = str(m).strip()
                if m not in index:
                    errors.append(f"{where}, {_cell(r, heads['material'])}: {m} is not a material of "
                                  f"the first sheet")
                    continue
                if m in seen:
                    errors.append(f"{where}, {_cell(r, heads['material'])}: {m} appears twice "
                                  f"(also row {seen[m] + 1})")
                    continue
                seen[m] = r
                col = row.get("color")
                if not blank(col):
                    col = str(col).strip()
                    if re.fullmatch(r"#[0-9a-fA-F]{6}", col):
                        colours[index[m]] = col.lower()
                    else:
                        errors.append(f"{where}, {_cell(r, heads['color'])} ({m}): {col!r} is not a "
                                      f"colour; write it as #rrggbb, e.g. #ff0000 for red")
                if not blank(row.get("label")):
                    labels[index[m]] = str(row["label"]).strip()
            missing = [n for n in names if n not in seen]
            if missing and not errors:
                warnings_.append(f"sheet {sheet} has no row for {', '.join(missing)}; "
                                 f"{'these use' if len(missing) > 1 else 'it uses'} the default colours")

    if len(names) > N_DEFAULT_COLOURS and all(c is None for c in colours):
        warnings_.append(f"{len(names)} materials and no colours given: beyond the first "
                         f"{N_DEFAULT_COLOURS}, colours are generated and may be hard to tell apart. "
                         f"Consider a {DISPLAY_SHEET} sheet")

    defaults = default_colours(len(names))
    colours = [c or d for c, d in zip(colours, defaults)]
    labels = [lb or n for lb, n in zip(labels, names)]
    if not errors:
        for what, values in (("colour", colours), ("label", labels)):
            groups = {}
            for n, v in zip(names, values):
                groups.setdefault("".join(v.split()).lower(), []).append(n)
            for v, ms in groups.items():
                if len(ms) > 1:
                    errors.append(f"sheet {sheet or DISPLAY_SHEET}: {', '.join(ms)} have the same {what}, "
                                  f"{v}; each material needs its own")
    return colours, labels, errors, warnings_


def _cell(r, c):
    from openpyxl.utils import get_column_letter
    return f"{get_column_letter(c + 1)}{r + 1}"


def material_display(path, names=None):
    """(colours, labels): the colour (#rrggbb) and the label each material of the database is
    shown with, in the order of the first sheet, or of names when given (a run's materials).
    ValueError when the Display sheet has errors."""
    if names is None:
        names = [str(m).strip() for m in _read(path).iloc[2:, 0].tolist()]
    names = [str(m).strip() for m in names]
    colours, labels, errors, _ = _display(path, names)
    if errors:
        raise ValueError(f"{path}:\n  " + "\n  ".join(errors))
    return colours, labels


# ---------------------------------------------------------------------- the Excel template
def write_database(path, attributes, units, names, labels, colours, values):
    """A whole database, as the app's editor saves it: the template's layout and Excel checks
    (write_template), with each material's name, the numbers, each column's unit, and on the Display
    sheet each material's colour and label (None: shown by its name)."""
    from openpyxl import load_workbook
    from openpyxl.styles import PatternFill
    write_template(path, attributes, len(names))
    wb = load_workbook(path)
    main, disp = wb["Main"], wb[DISPLAY_SHEET]
    for c, unit in enumerate(units, start=2):
        main.cell(row=2, column=c, value=unit)
    for r, (name, row) in enumerate(zip(names, values), start=3):
        main.cell(row=r, column=1, value=name)
        for c, v in enumerate(row, start=2):
            main.cell(row=r, column=c, value=float(v))
    for r, (colour, label) in enumerate(zip(colours, labels), start=2):
        disp.cell(row=r, column=2, value=colour)
        disp.cell(row=r, column=2).fill = PatternFill("solid", fgColor=colour[1:])
        disp.cell(row=r, column=3, value=label or None)
    wb.save(path)
    return path


def write_template(path, attributes, n_materials=5):
    """An empty database to fill in: the attributes in row 1, a units list under each in
    row 2, numbers only below; a Display sheet with each material's default colour and an
    empty label; and a Reference sheet describing every known attribute."""
    from openpyxl import Workbook
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation
    names = [attribute_name(a) for a in attributes]
    unknown = [a for a, n in zip(attributes, names) if n not in ATTRIBUTES]
    if unknown:
        raise ValueError(f"not known attributes: {', '.join(unknown)}; known: {', '.join(ATTRIBUTES)}")
    wb = Workbook()
    ws = wb.active
    ws.title = "Main"
    ws["A1"], ws["A2"] = "Attributes", "Material/Units"
    last = 2 + n_materials
    for c, name in enumerate(names, start=2):
        col, a = get_column_letter(c), ATTRIBUTES[name]
        ws[f"{col}1"], ws[f"{col}2"] = name, a.units[0]
        units = DataValidation(type="list", formula1='"' + ",".join(a.units) + '"', allow_blank=False)
        units.error = f"Choose one of: {', '.join(a.units)}"
        ws.add_data_validation(units)
        units.add(f"{col}2")
        first = f"{col}3"                       # Excel shifts the formula down the column
        conditions = [f"ISNUMBER({first})"]
        if a.low is not None:
            conditions.append(f"{first}{'>=' if 'low' in a.closed else '>'}{a.low}")
        if a.high is not None:
            conditions.append(f"{first}{'<=' if 'high' in a.closed else '<'}{a.high}")
        number = DataValidation(type="custom", formula1=f"AND({','.join(conditions)})")
        number.error = f"{name}: a number, {a.range_text()}"
        ws.add_data_validation(number)
        number.add(f"{col}3:{col}{last}")
        ws.column_dimensions[col].width = max(14, len(name) + 2)
    for r in range(3, last + 1):
        ws[f"A{r}"] = f"Material{r - 2}"
    ws.column_dimensions["A"].width = 16

    from openpyxl.styles import PatternFill
    disp = wb.create_sheet(DISPLAY_SHEET)
    disp.append(["Material", "Color", "Label"])
    for r, colour in enumerate(default_colours(n_materials), start=2):
        disp.append([f"=Main!A{r + 1}", colour, None])         # follows the name typed on Main
        disp[f"B{r}"].fill = PatternFill("solid", fgColor=colour[1:])  # the default; not updated by edits
    hexa = DataValidation(type="custom", formula1='AND(LEN(B2)=7,LEFT(B2,1)="#",ISNUMBER(HEX2DEC(MID(B2,2,6))))')
    hexa.error = "A colour as #rrggbb, e.g. #ff0000 for red"
    disp.add_data_validation(hexa)
    hexa.add(f"B2:B{n_materials + 1}")
    for col, w in zip("ABC", (16, 12, 16)):
        disp.column_dimensions[col].width = w

    ref = wb.create_sheet("Reference")
    ref.append(["Attribute", "Meaning", "Allowed range", "Units"])
    for a in ATTRIBUTES.values():
        ref.append([a.name, a.doc, a.range_text(), ", ".join(a.units)])
    for col, w in zip("ABCD", (18, 52, 22, 36)):
        ref.column_dimensions[col].width = w
    wb.save(path)
    return path
