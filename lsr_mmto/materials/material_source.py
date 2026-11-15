"""
material_source.py
==================
Where a run gets each element's material properties from. A material source turns each element's latent
point z into its material's properties with decode(z): {column name: one value per element}, every column
of the database. The run, the final check, the history and the save ask the source; none of them reaches
into the material network itself (tests/code_rules/test_material_source.py checks it).

The sources:
    MaterialEncoder     materials/encoder.py: the material network. z through its decoder, then back to each
                        column's numbers, inside the torch tree, so the gradient reaches z
    DatabaseMaterials   this file: each element one material of the database, at its database values, with
                        no network. Analyse uses it (optimize/solid_part_analysis.py)

What every source gives (MaterialSource): decode(z); network_weights(), the weights a run saves, None with no
network; and the database it reads: materialNames, rawData (materials, columns) and materialAttributes
({column: {"idx": its column in rawData, ...}}).
"""
from typing import Any, Protocol

import numpy as np
import torch

_DTYPE = torch.float64


class MaterialSource(Protocol):
    """What the run, the final check, the history and the save read of a material source."""

    materialNames: list[str]
    rawData: np.ndarray
    materialAttributes: dict[str, dict[str, Any]]

    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Each point's properties, {column: (points,)}, from latent points z (points, latent dimension)."""

    def network_weights(self) -> dict[str, torch.Tensor] | None:
        """The weights of the network a run saves with it; None for a source with no network."""


class DatabaseMaterials:
    """The database's materials, each element given one of them: rows, its row in the database, one per element.
    decode gives every element its material's database values, whatever the latent points, so a quantity of a
    database material (Mass fraction) reads the database, and the final check's mass is the materials' own."""

    materialNames: list[str]
    rawData: np.ndarray                  # the database's numbers, (materials, attributes)
    rows: np.ndarray                     # each element's material, its row in the database
    materialAttributes: dict[str, dict[str, Any]]   # {column: {"idx", "unit"}}

    def __init__(self, xlsx: str, rows: np.ndarray) -> None:
        from lsr_mmto.materials.databases import database_table
        names, attributes, units, values = database_table(xlsx)
        self.materialNames, self.rawData, self.rows = names, values, np.asarray(rows, dtype=int)
        self.materialAttributes = {a: {"idx": j, "unit": units[j]} for j, a in enumerate(attributes)}

    def properties(self, n: int) -> dict[str, torch.Tensor]:
        """{column: each of the first n elements' material's value}, as torch, the way a run's decode gives them."""
        return {a: torch.as_tensor(self.rawData[self.rows[:int(n)], c["idx"]], dtype=_DTYPE)
                for a, c in self.materialAttributes.items()}

    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """properties(len(z)): the first len(z) elements' materials' values; z's values are not read."""
        return self.properties(len(z))

    def network_weights(self) -> None:
        """None: there is no network."""
        return None
