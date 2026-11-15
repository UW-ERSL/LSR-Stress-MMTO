"""
network_stamp.py
================
The record a trained network keeps of the database it learned from, so that a run never uses a network
on a database that has changed since. When a network is trained, every cell of the database's first sheet,
exactly as stored, and each material's colour and label go into the network's file with its weights. Before
a run the database is read again and compared with that record:

    SAME          nothing changed: the network is used
    CHANGED       a cell of the first sheet changed (a number, a unit, a name, a material added or taken
                  away): the run does not start until a new network is trained or the version the network
                  was trained on is brought back
    DISPLAY       only colours or labels changed: the run asks once to go on
    TO_CONFIRM    the database changed in a way the network still fits (its units converted); asked once
    NO_STAMP      a network trained before networks kept this record: asked once whether it was trained on
                  the database as it is now

A Yes to the question stamps the network with the database as it is now (stamp), so it is not asked again.

    database_stamp(xlsx, sheet)     the record of a database as it is now
    read_network(nt)                (weights, stamp) of a network file; stamp None when it has none
    write_network(nt, weights, record)   a network file written with its weights and a record
    stamp(nt, xlsx, to_confirm)     a network stamped again with the database as it is now
    compare(xlsx, nt)               a Comparison: one of the states above and a line saying what changed
    changed_cells(then, now), listed(lines)   the cells two records differ in, and those cells in a line
"""
import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

SAME, CHANGED, DISPLAY, TO_CONFIRM, NO_STAMP = "same", "changed", "display", "to confirm", "no stamp"
CONFIRMABLE = (DISPLAY, TO_CONFIRM, NO_STAMP)          # asked once before a run; a yes stamps the network again
SHOWN_CELLS = 3                                         # changed cells a line names before "and N more"


def _cell_text(v: Any) -> str:
    """A cell as the record keeps it: a number as the exact float it is stored as (5 and 5.0 are the same
    number), text as it is, an empty cell as ""."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, bool):
        return str(v)
    try:
        import numpy as np
        if isinstance(v, (int, float, np.integer, np.floating)):
            return repr(float(v))
    except ImportError:
        if isinstance(v, (int, float)):
            return repr(float(v))
    return str(v)


def database_stamp(xlsx: str, sheet: "pd.DataFrame | None" = None) -> dict[str, Any]:
    """The record of the database xlsx as it is now: {"main": every cell of its first sheet, row by row,
    trailing empty rows and columns left out; "display": [colour, label] of each material, or the Display
    sheet's error}. The first sheet is sheet when given (the one a network is trained on, as it was read),
    else read from the file, never from what was read before."""
    import pandas as pd
    from lsr_mmto.materials.databases import material_display
    df = sheet
    if df is None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Data Validation extension")
            df = pd.read_excel(xlsx, header=None)
    main = [[_cell_text(v) for v in row] for row in df.itertuples(index=False)]
    while main and not any(main[-1]):
        main.pop()
    width = max((max((i + 1 for i, t in enumerate(row) if t), default=0) for row in main), default=0)
    main = [row[:width] + [""] * (width - len(row)) for row in main]
    try:
        colours, labels = material_display(xlsx)
        display: Any = [[c, lb] for c, lb in zip(colours, labels)]
    except ValueError as e:
        display = str(e)
    return {"main": main, "display": display}


def read_network(nt: str) -> "tuple[dict[str, Any], dict[str, Any] | None]":
    """(weights, stamp) of the network file nt; stamp None for a network saved before networks kept one."""
    import torch
    saved = torch.load(nt, map_location="cpu", weights_only=True)
    if isinstance(saved, dict) and "weights" in saved and isinstance(saved["weights"], dict):
        return saved["weights"], saved.get("database")
    return saved, None


def write_network(nt: str, weights: dict[str, Any], record: dict[str, Any]) -> None:
    """The network file nt written: its weights and record, the database it was trained on (database_stamp).
    Written beside it first and then put in its place, so a write cut short never leaves half a network."""
    import os
    import torch
    part = nt + ".part"
    torch.save({"weights": weights, "database": record}, part)
    os.replace(part, nt)


def stamp(nt: str, xlsx: str, to_confirm: str | None = None) -> None:
    """The network nt stamped again with the database xlsx as it is now (a Yes to the question compare asks);
    to_confirm, when given, says why the next run asks once before using it. ValueError for a network that comes
    with LSR MMTO, which is never changed."""
    from lsr_mmto.materials.databases import in_shipped_folder
    if in_shipped_folder(nt):
        raise ValueError(f"{nt} comes with LSR MMTO and is never changed")
    weights, _ = read_network(nt)
    write_network(nt, weights, dict(database_stamp(xlsx), to_confirm=to_confirm))


@dataclass(frozen=True)
class Comparison:
    """state: SAME, CHANGED, DISPLAY, TO_CONFIRM or NO_STAMP; text: what changed, in a sentence ("" for SAME)."""
    state: str
    text: str


def changed_cells(then: list[list[str]], now: list[list[str]]) -> list[str]:
    """Each cell of the first sheet that differs between two records' "main", as "cell B4 is x, was y"."""
    from lsr_mmto.materials.databases import _cell
    out = []
    for r in range(max(len(then), len(now))):
        a = then[r] if r < len(then) else []
        b = now[r] if r < len(now) else []
        for c in range(max(len(a), len(b))):
            was, is_ = (a[c] if c < len(a) else ""), (b[c] if c < len(b) else "")
            if was != is_:
                out.append(f"cell {_cell(r, c)} is {is_ or 'empty'}, was {was or 'empty'}")
    return out


def listed(lines: list[str]) -> str:
    """The first SHOWN_CELLS lines, then how many more."""
    shown = "; ".join(lines[:SHOWN_CELLS])
    return shown + (f"; and {len(lines) - SHOWN_CELLS} more" if len(lines) > SHOWN_CELLS else "")


def compare(xlsx: str, nt: str) -> Comparison:
    """The database xlsx as it is now against the record in the network nt."""
    _, then = read_network(nt)
    if then is None:
        return Comparison(NO_STAMP, "it was trained before networks kept a record of their database, so "
                                    "it can't be checked against the database")
    now = database_stamp(xlsx)
    cells = changed_cells(then.get("main", []), now["main"])
    if cells:
        return Comparison(CHANGED, f"the database has changed since its network was trained: {listed(cells)}")
    if then.get("display") != now["display"]:
        was, is_ = then.get("display"), now["display"]
        names = [row[0] for row in now["main"][2:]]
        if isinstance(was, list) and isinstance(is_, list) and len(was) == len(is_) == len(names):
            lines = [f"{n}: {what} {b}, was {a}" for n, old, new in zip(names, was, is_)
                     for what, a, b in (("colour", old[0], new[0]), ("label", old[1], new[1])) if a != b]
        else:
            lines = ["the Display sheet"]
        return Comparison(DISPLAY, f"only colours or labels changed since its network was trained: {listed(lines)}")
    if then.get("to_confirm"):
        return Comparison(TO_CONFIRM, str(then["to_confirm"]))
    return Comparison(SAME, "")
