"""
test_stl_export.py
==================
The per-material STL export (utils/stl_export.py) on shapes whose answer is known: three slabs
around a spherical hole, a plate one element thick (as the L-bracket and the U-bracket), a sheet of
one material one element thick between two others, scattered specks, a round cylinder. Every piece
is closed, touches no other body at a single point, and the pieces add up exactly to their whole;
volumes are near the design's, and the cylinder comes out round. Contacts along an edge or at a
corner are always resolved. About half a minute.

    python tests/test_stl_export.py        (or: python -m pytest tests)
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils import project_paths  # noqa: E402

project_paths.bootstrap()

from utils import stl_export as X  # noqa: E402


def _design(labels):
    """A run and a mesh with one element per cell of labels (0 void, m + 1 material m), density 0 or 1."""
    ijk = np.argwhere(labels > 0)
    mesh = SimpleNamespace(elem_centers=ijk + 0.5, elem_size=np.ones(3))
    lab = labels[tuple(ijk.T)]
    n = int(labels.max())
    run = SimpleNamespace(x_cont=np.ones(len(ijk)), material_indices=lab - 1, solid=np.ones(len(ijk), bool),
                          material_labels=[f"m{i}" for i in range(n)])
    return run, mesh


def _point_contacts(t):
    """Bodies that share a vertex but no edge (edge-connected bodies minus vertex-connected bodies)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    f = np.asarray(t.faces)
    rows = np.repeat(np.arange(len(f)), 3)
    g = coo_matrix((np.ones(len(rows)), (rows, f.ravel())), shape=(len(f), len(t.vertices))).tocsr()
    by_vertex = connected_components(g @ g.T, directed=False)[0]
    return len(t.split(only_watertight=False)) - by_vertex


def _pieces(labels):
    run, mesh = _design(labels)
    label, origin, spacing = X.material_labels(run, mesh)
    pts, q, bl, behind = X.material_surfaces(label, origin, spacing)
    whole = X.piece(pts, q, bl, behind, 0)
    out = {m: X.piece(pts, q, bl, behind, m) for m in sorted(set(np.unique(label)) - {0})}
    return out, whole


def _check(labels, tolerance, bodies=None):
    out, whole = _pieces(labels)
    assert whole.is_watertight
    assert abs(sum(p.volume for p in out.values()) / whole.volume - 1) < 1e-6        # no gap, no overlap
    for m, p in out.items():
        exact = float(np.count_nonzero(labels == m))
        assert p.is_watertight and _point_contacts(p) == 0, m
        assert abs(p.volume / exact - 1) < tolerance, (m, p.volume, exact)
        if bodies and m in bodies:
            assert len(p.split(only_watertight=False)) == bodies[m], m
    return out


def test_slabs_and_a_hole():
    L = np.zeros((60, 30, 24), int)
    L[:20], L[20:40], L[40:] = 1, 2, 3
    x, y, z = np.meshgrid(np.arange(60) + .5, np.arange(30) + .5, np.arange(24) + .5, indexing="ij")
    L[(x - 40) ** 2 + (y - 15) ** 2 + (z - 12) ** 2 < 64] = 0
    _check(L, 0.02, bodies={1: 1, 2: 1, 3: 1})


def test_plate_one_element_thick():
    P = np.zeros((80, 60, 1), int)
    P[:, :20], P[:, 20:40], P[:, 40:] = 1, 2, 3
    x, y = np.meshgrid(np.arange(80) + .5, np.arange(60) + .5, indexing="ij")
    P[(x - 40) ** 2 + (y - 30) ** 2 < 100, 0] = 0
    _check(P, 0.03, bodies={1: 1, 2: 2, 3: 1})              # the hole cuts the middle band in two


def test_sheet_one_element_thick():
    S = np.ones((40, 30, 20), int)
    S[20:] = 3
    S[20] = 2
    _check(S, 0.01)


def test_specks_and_contacts():
    for fraction in (0.02, 0.10):
        rng = np.random.default_rng(0)
        C = np.ones((40, 30, 20), int)
        C[rng.random(C.shape) < fraction] = 2
        C[30:] = 3
        out, whole = _pieces(C)
        assert abs(sum(p.volume for p in out.values()) / whole.volume - 1) < 1e-6
        assert all(p.is_watertight and _point_contacts(p) == 0 for p in out.values()), fraction
        for m in (1, 3):                                    # the big materials keep their volume; specks
            assert abs(out[m].volume / np.count_nonzero(C == m) - 1) < 0.15, (fraction, m)   # one element
    rng = np.random.default_rng(1)                          # across are below the smoothing width
    C = np.ones((40, 30, 20), int)
    C[rng.random(C.shape) < 0.25] = 2
    L = np.pad(np.repeat(np.repeat(np.repeat(C, 3, 0), 3, 1), 3, 2), 1)
    joined, changed = X.contacts_joined(L)
    assert changed > 0 and X.contacts_joined(joined, max_rounds=1)[1] == 0   # all resolved, and it stopped


def test_round_cylinder():
    r0, n = 6.0, (30, 30, 10)
    x, y = np.meshgrid(np.arange(n[0]) + .5, np.arange(n[1]) + .5, indexing="ij")
    Cy = np.zeros(n, int)
    Cy[(x - 15) ** 2 + (y - 15) ** 2 < r0 ** 2, :] = 1
    out, _ = _pieces(Cy)
    v = out[1].vertices
    side = (v[:, 2] > 0.5) & (v[:, 2] < n[2] - 0.5)
    deviation = np.abs(np.hypot(v[side, 0] - 15, v[side, 1] - 15) - r0)
    assert abs(out[1].volume / (np.pi * r0 ** 2 * n[2]) - 1) < 0.02
    assert deviation.mean() < 0.15 and deviation.max() < 0.5       # in elements: round, not a staircase


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
