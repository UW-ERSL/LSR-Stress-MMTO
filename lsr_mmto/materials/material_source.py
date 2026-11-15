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
    PinnedMaterial      this file: the network pinned to one database material: every element that material's
                        properties as the network gives them, at its own latent point; no latent points
    DatabaseMaterials   this file: each element one material of the database, at its database values, with
                        no network. Analyse uses it (optimize/solid_part_analysis.py), and so does a project of
                        one database material (Project.material_source)

What every source gives (MaterialSource): latent_dim, the numbers in each element's latent point (0 with no
latent points: the design is then the densities only); decode(z); the database material nearest each latent
point (getClosestRealMaterialIndex) and that material's own latent point (getClosestRealMaterialZValues);
the latent points of the lightest and the heaviest material (getLightestMaterial, getHeaviestMaterial), a
run's start; material_properties(), each database material's properties as the source gives them, which a
run saves; network_weights() and reconstruction(), what a run saves of its network (None and [] with no
network); and the database it reads: materialNames, rawData (materials, columns) and materialAttributes
({column: {"idx": its column in rawData, ...}}).

The network's materials as latent points, and its errors, come from where the run makes its source
(optimize/optimization_loop.py, material_source).
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
    latent_dim: int

    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Each point's properties, {column: (points,)}, from latent points z (points, latent dimension)."""

    def getClosestRealMaterialIndex(self, zDesign: torch.Tensor) -> torch.Tensor:
        """The database row of the material nearest each point, (points,)."""

    def getClosestRealMaterialZValues(self, zDesign: torch.Tensor) -> np.ndarray:
        """The latent point of the material nearest each point, (points, latent dimension)."""

    def getLightestMaterial(self) -> np.ndarray:
        """The latent point (latent dimension,) of the lightest material."""

    def getHeaviestMaterial(self) -> np.ndarray:
        """The latent point (latent dimension,) of the heaviest material."""

    def material_properties(self) -> dict[str, torch.Tensor]:
        """Each database material's properties as this source gives them, {column: (materials,)}."""

    def network_weights(self) -> dict[str, torch.Tensor] | None:
        """The weights of the network a run saves with it; None for a source with no network."""

    def reconstruction(self) -> list[dict[str, Any]]:
        """Each material's every property as the database gives it and as the network gives it back; [] for a
        source with no network."""


class PinnedMaterial:
    """The network pinned to one database material, row: every element gets that material's properties as the
    network gives them, at the material's own latent point (the network's training_latents). It has no latent
    points (latent_dim 0), so the design is the densities only; what a run saves of its network is the
    network's."""

    latent_dim = 0
    network: Any                         # the MaterialEncoder, read or trained
    row: int                             # the material's row in the database
    materialNames: list[str]
    rawData: np.ndarray
    materialAttributes: dict[str, dict[str, Any]]

    def __init__(self, network: Any, row: int) -> None:
        self.network, self.row = network, int(row)
        self.materialNames, self.rawData = network.materialNames, network.rawData
        self.materialAttributes = network.materialAttributes
        z = torch.as_tensor(network.training_latents[self.row:self.row + 1], dtype=_DTYPE)
        with torch.no_grad():
            self._pinned = {k: v.detach() for k, v in network.decode(z).items()}     # each column, (1,)

    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """The pinned material's properties for each of len(z) elements; z's values are not read."""
        return {k: v.repeat(len(z)) for k, v in self._pinned.items()}

    def getClosestRealMaterialIndex(self, zDesign: torch.Tensor) -> torch.Tensor:
        """The pinned material's row, for every point."""
        return torch.full((len(zDesign),), self.row, dtype=torch.long)

    def getClosestRealMaterialZValues(self, zDesign: torch.Tensor) -> np.ndarray:
        """(points, 0): there are no latent points."""
        return np.zeros((len(zDesign), 0))

    def getLightestMaterial(self) -> np.ndarray:
        """(0,): there are no latent points."""
        return np.zeros(0)

    getHeaviestMaterial = getLightestMaterial

    def material_properties(self) -> dict[str, torch.Tensor]:
        """Each database material's properties as the network gives them."""
        return self.network.material_properties()

    def network_weights(self) -> dict[str, torch.Tensor]:
        """The network's weights."""
        return self.network.network_weights()

    def reconstruction(self) -> list[dict[str, Any]]:
        """The network's record of each material, the database's values against its own."""
        return self.network.reconstruction()


class DatabaseMaterials:
    """The database's materials, each element given one of them: rows, its row in the database, one per element,
    or one row for every element.
    decode gives every element its material's database values, whatever the latent points, so a quantity of a
    database material (Mass fraction) reads the database, and the final check's mass is the materials' own.
    It has no latent points (latent_dim 0) and no network: the design is the densities only."""

    latent_dim = 0

    materialNames: list[str]
    rawData: np.ndarray                  # the database's numbers, (materials, attributes)
    rows: np.ndarray                     # each element's material, its row in the database; or one row, for all
    materialAttributes: dict[str, dict[str, Any]]   # {column: {"idx", "unit"}}

    def __init__(self, xlsx: str, rows: np.ndarray | int) -> None:
        from lsr_mmto.materials.databases import database_table
        names, attributes, units, values = database_table(xlsx)
        self.materialNames, self.rawData, self.rows = names, values, np.asarray(rows, dtype=int)
        self.materialAttributes = {a: {"idx": j, "unit": units[j]} for j, a in enumerate(attributes)}

    def _rows(self, n: int) -> np.ndarray:
        """The first n elements' rows."""
        return self.rows[:int(n)] if self.rows.ndim else np.full(int(n), int(self.rows))

    def properties(self, n: int) -> dict[str, torch.Tensor]:
        """{column: each of the first n elements' material's value}, as torch, the way a run's decode gives them."""
        return {a: torch.as_tensor(self.rawData[self._rows(n), c["idx"]], dtype=_DTYPE)
                for a, c in self.materialAttributes.items()}

    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """properties(len(z)): the first len(z) elements' materials' values; z's values are not read."""
        return self.properties(len(z))

    def getClosestRealMaterialIndex(self, zDesign: torch.Tensor) -> torch.Tensor:
        """Each of the first len(zDesign) elements' own row: its material is the one it was given."""
        return torch.as_tensor(self._rows(len(zDesign)))

    def getClosestRealMaterialZValues(self, zDesign: torch.Tensor) -> np.ndarray:
        """(points, 0): there are no latent points."""
        return np.zeros((len(zDesign), 0))

    def getLightestMaterial(self) -> np.ndarray:
        """(0,): there are no latent points."""
        return np.zeros(0)

    getHeaviestMaterial = getLightestMaterial

    def material_properties(self) -> dict[str, torch.Tensor]:
        """Each database material's own values, {column: (materials,)}."""
        return {a: torch.as_tensor(self.rawData[:, c["idx"]], dtype=_DTYPE) for a, c in self.materialAttributes.items()}

    def network_weights(self) -> None:
        """None: there is no network."""
        return None

    def reconstruction(self) -> list[dict[str, Any]]:
        """[]: there is no network to compare with the database."""
        return []
