"""
test_material_source.py
=======================
A run gets each element's material properties from its material source (lsr_mmto/materials/material_source.py)
and from nothing else: outside the material network's own file and the material source's, no code reaches
into the network's decoder. Every source gives what the run reads (MaterialSource), and each decodes as it
says: the network inside the torch tree, the database at each element's material's own values.

    python -m pytest tests/code_rules/test_material_source.py
"""
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from lsr_mmto.materials.material_source import DatabaseMaterials, MaterialSource  # noqa: E402

DATABASE = os.path.join(ROOT, "data", "databases", "6Materials_AF_3Attr")
OWN_FILES = {os.path.join("lsr_mmto", "materials", "encoder.py"),
             os.path.join("lsr_mmto", "materials", "material_source.py")}


def _network():
    """The 6-material database's material encoder, with its shipped network."""
    from lsr_mmto.materials.databases import DATABASE_VAE, VAEParams
    from lsr_mmto.materials.encoder import MaterialEncoder
    params = VAEParams()
    for k, v in DATABASE_VAE["6Materials_AF_3Attr"].items():
        setattr(params, k, v)
    enc = MaterialEncoder(params)
    enc.readExcel(DATABASE + ".xlsx")
    enc.loadAutoencoderFromFile(DATABASE + ".nt")
    return enc


def test_nothing_but_the_material_source_reaches_into_the_network():
    """No file of lsr_mmto or gui, other than the network's own (encoder.py) and the material source's, calls the
    network's decoder or turns its output into properties itself: each asks its material source's decode."""
    wrong = []
    for path in sorted(glob.glob(os.path.join(ROOT, "lsr_mmto", "**", "*.py"), recursive=True) +
                       glob.glob(os.path.join(ROOT, "gui", "**", "*.py"), recursive=True)):
        if os.path.relpath(path, ROOT) in OWN_FILES:
            continue
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                if "vaeNet.decoder" in line or "getMaterialProperties(" in line:
                    wrong.append(f"{os.path.relpath(path, ROOT)}:{i}: {line.strip()}")
    assert not wrong, "ask the material source's decode instead:\n" + "\n".join(wrong)


def test_every_source_gives_what_the_run_reads():
    """The network, the network pinned to one material and the database's own values each give every member of
    MaterialSource."""
    from lsr_mmto.materials.encoder import MaterialEncoder
    from lsr_mmto.materials.material_source import PinnedMaterial
    members = [n for n in vars(MaterialSource) if not n.startswith("_")] + list(MaterialSource.__annotations__)
    for source in (MaterialEncoder, PinnedMaterial, DatabaseMaterials):
        missing = [n for n in members if not hasattr(source, n) and n not in getattr(source, "__annotations__", {})]
        assert not missing, (source.__name__, missing)


def test_the_network_decodes_inside_the_tree():
    """The network's decode gives every column of the database, one value per point, and the gradient of a
    property reaches the latent points."""
    enc = _network()
    z = torch.zeros((4, enc.vae_params.latentDim), dtype=torch.float64, requires_grad=True)
    props = enc.decode(z)
    assert sorted(props) == sorted(enc.materialAttributes)
    assert all(v.shape == (4,) for v in props.values())
    props["Youngs_Modulus"].sum().backward()
    assert z.grad is not None and torch.any(z.grad != 0)
    assert enc.network_weights().keys() == enc.vaeNet.state_dict().keys()


def test_the_database_decodes_to_each_elements_own_material():
    """The database's own values: each element gets its material's row, whatever its latent point, and there is no
    network to save."""
    rows = np.array([2, 0, 0, 5])
    source = DatabaseMaterials(DATABASE + ".xlsx", rows)
    props = source.decode(torch.full((4, 3), 7.0))
    for name, c in source.materialAttributes.items():
        assert np.allclose(props[name].numpy(), source.rawData[rows, c["idx"]]), name
    assert source.network_weights() is None


def test_every_quantity_and_penalty_says_truly_whether_it_reads_the_latent_points():
    """A quantity or a penalty whose own function reads the model's latent points (m.z, m.z_real) says so
    (reads_latent_points), and one that does not says it does not: a single-material project refuses the first
    kind of quantity, and a run with no latent points leaves out the first kind of penalty, so a wrong word would
    let a value of no meaning into a run. Read from the names the function's own code uses."""
    from lsr_mmto.design.latent_penalties import PENALTIES
    from lsr_mmto.formulation.quantities import QUANTITIES
    wrong = [f"{q.name}: says {q.reads_latent_points}" for q in list(QUANTITIES) + list(PENALTIES)
             if bool({"z", "z_real"} & set(q.fn.__code__.co_names)) != q.reads_latent_points]
    assert not wrong, wrong
