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


LATENT = {"z", "z_real"}                       # the model's latent points, as its code names them
OURS = [os.path.normcase(os.path.join(ROOT, d)) for d in ("lsr_mmto", "tests")]   # code this check follows


def _ours(fn):
    """fn is a plain function, or a module, written in LSR MMTO (its package or its tests), not a library's."""
    import types
    if isinstance(fn, types.FunctionType):
        path = fn.__code__.co_filename
    elif isinstance(fn, types.ModuleType):
        path = getattr(fn, "__file__", None) or ""
    else:
        return False
    path = os.path.normcase(os.path.abspath(path))
    return any(_inside(path, base) for base in OURS)


def _inside(path, folder):
    """path is folder or inside it; False for a path on another drive (Windows), which commonpath refuses."""
    try:
        return os.path.commonpath([path, folder]) == folder
    except ValueError:
        return False


def _unwrapped(g):
    """The plain function behind g: a method's, a partial's, a decorated function's, a callable object's."""
    import functools
    for _ in range(10):
        if isinstance(g, functools.partial):
            g = g.func
        elif hasattr(g, "__wrapped__"):
            g = g.__wrapped__
        elif hasattr(g, "__func__"):
            g = g.__func__
        elif not callable(g) or isinstance(g, type) or hasattr(g, "__code__"):
            return g
        else:
            g = type(g).__call__
    return g


def reaches_the_latent_points(fn, seen=None):
    """The chain of functions by which fn reads the latent points (z, z_real), or None: read from the names in
    its code (an attribute, or the text "z" as getattr reads it), and in the code of every function of LSR MMTO
    it calls, however deep: by name, as a module's function (module.helper), as a method of the model
    (m.<method>), imported inside it, held in a closure or a default, through a partial or a decorator; lambdas
    and functions defined inside included. Not followed: a method of a class other than the model. It errs on
    the side of a read: any attribute named z counts (and the text "z" where getattr is called), and any name
    of a model method is followed into it."""
    import types
    from lsr_mmto.optimize.design_model import MMTOModelAD
    seen = set() if seen is None else seen
    f = _unwrapped(fn)
    if not isinstance(f, types.FunctionType) or f.__code__ in seen:
        return None
    seen.add(f.__code__)
    called = [c.cell_contents for c in (f.__closure__ or ()) if _cell_full(c)] + list(f.__defaults__ or ())
    stack = [f.__code__]
    while stack:
        code = stack.pop()
        stack += [c for c in code.co_consts if isinstance(c, types.CodeType)]
        texts = {c for c in code.co_consts if isinstance(c, str)} if "getattr" in code.co_names else set()
        if LATENT & (set(code.co_names) | texts):
            return [f.__qualname__]
        for name in code.co_names:
            g = f.__globals__.get(name, sys.modules.get(name))     # a module imported inside: by its full name
            called += [g, getattr(MMTOModelAD, name, None)]
            called += [getattr(g, other, None) for other in code.co_names if _ours(g)]
    for g in called:
        chain = reaches_the_latent_points(g, seen) if _ours(_unwrapped(g)) else None
        if chain:
            return [f.__qualname__] + chain
    return None


def _cell_full(cell):
    try:
        cell.cell_contents
        return True
    except ValueError:                          # a closure variable not yet given a value
        return False


def test_every_quantity_and_penalty_says_truly_whether_it_reads_the_latent_points():
    """A quantity or a penalty whose own function reads the model's latent points (m.z, m.z_real) says so
    (reads_latent_points), and one that does not says it does not: a single-material project refuses the first
    kind of quantity, and a run with no latent points leaves out the first kind of penalty, so a wrong word would
    let a value of no meaning into a run. Read from the names the function's code uses, and the code of every
    function of LSR MMTO it calls, however deep (reaches_the_latent_points), so a quantity of any physics that
    reads them through a helper is caught too."""
    from lsr_mmto.design.latent_penalties import PENALTIES
    from lsr_mmto.formulation.quantities import QUANTITIES
    wrong = [f"{q.name}: says {q.reads_latent_points}, but " + (" -> ".join(chain) + " reads them" if chain else
                                                                  "nothing it calls reads them")
             for q in list(QUANTITIES) + list(PENALTIES)
             for chain in [reaches_the_latent_points(q.fn)] if bool(chain) != q.reads_latent_points]
    assert not wrong, wrong


def _reads_through_a_helper(m, opt):
    return _helper_of_a_helper(m) * m.elem_volume


def _helper_of_a_helper(m):
    return m.z.sum()


def test_the_check_follows_a_function_into_its_helpers():
    """reaches_the_latent_points finds the latent points read two helpers down, in a lambda, through a module's
    function, a closure, a partial, a decorator, a callable object and getattr(m, "z"); a quantity's own
    function that reads none (Mass) gives None."""
    import functools
    import sys as system
    from lsr_mmto.formulation.quantities import QUANTITIES
    here = system.modules[__name__]
    assert reaches_the_latent_points(_reads_through_a_helper) == ["_reads_through_a_helper", "_helper_of_a_helper"]
    assert reaches_the_latent_points(lambda m: _helper_of_a_helper(m)) is not None
    assert reaches_the_latent_points(lambda m: here._helper_of_a_helper(m)) is not None
    held = _helper_of_a_helper
    assert reaches_the_latent_points(lambda m: held(m)) is not None
    assert reaches_the_latent_points(functools.partial(_reads_through_a_helper, opt=None)) is not None
    assert reaches_the_latent_points(functools.wraps(_helper_of_a_helper)(lambda m: 0)) is not None

    class Reader:
        def __call__(self, m):
            return m.z_real.sum()
    assert reaches_the_latent_points(Reader()) is not None
    assert reaches_the_latent_points(lambda m: getattr(m, "z").sum()) is not None
    assert reaches_the_latent_points(lambda m, read=_helper_of_a_helper: read(m)) is not None

    def imports_it_inside(m):
        import tests.code_rules.test_material_source as here_too
        return here_too._helper_of_a_helper(m)
    assert reaches_the_latent_points(imports_it_inside) is not None
    assert reaches_the_latent_points(lambda m: m.mesh.coordinates(axis="z")) is None
    assert reaches_the_latent_points(next(q for q in QUANTITIES if q.name == "Mass").fn) is None
