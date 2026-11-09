"""lsr_mmto/materials/pca.py gives what scikit-learn's PCA (1.9.1) gave: the reference numbers below were
worked out with it, before it was taken out of the requirements.

- every shipped database: d_eff, the eigenvalues and the explained variance ratios (participation_ratio);
- a 2D projection of 5D points, signs included (the latent views above three dimensions).
"""
import glob
import os

import numpy as np

from lsr_mmto.materials import databases as D
from lsr_mmto.materials.pca import PCA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# name: (d_eff, eigenvalues, ratios), from scikit-learn's PCA().fit on the normalized database
SHIPPED = {
    "6Materials_AF_3Attr": (1.5142150803463625, [0.32894536560932996, 0.09105463439038572, 1.310850526505944e-33],
                            [0.7832032514513158, 0.21679674854868417, 3.121072682159122e-33]),
    "BM1_3Mats_2Attr": (1.1395348837218393, [0.5451151054163976, 0.03821822790360216],
                        [0.9344830378780415, 0.0655169621219584]),
    "Beam_8Mats_3Attr_v4_spectrum2D": (1.9429374049298849,
                                       [0.21643961078341856, 0.14164079596414952, 0.0024893139555902852],
                                       [0.6002711774059476, 0.3928249873226497, 0.006903835271402701]),
    "CustomAlloys20_3D": (2.59822721366884,
                          [0.2025741147019636, 0.14793605258022993, 0.05886086114178185, 0.006057817133967455],
                          [0.487626501789725, 0.3561044307877659, 0.1416869862821601, 0.014582081140349047]),
}

# scikit-learn's PCA(n_components=2).fit(Z).transform(Y), Z and Y drawn as below
PROJECTED = [[0.327469432216962, -0.6029554569663502], [0.03358131667528773, 0.8555349449981426],
             [0.22699514558449685, -0.11582235726085496], [1.3776815148454176, 0.18803640000526647],
             [-1.7385984425854095, 0.24922847434076945], [-1.0399064724298424, -1.0674741424357748]]


def test_every_shipped_database_as_before():
    shipped = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(ROOT, "data", "databases", "*.xlsx")))
    assert shipped == sorted(SHIPPED), f"a database was added or removed: add its numbers here ({shipped})"
    for name, (d_eff, eigenvalues, ratios) in SHIPPED.items():
        got = D.database_pca(os.path.join(ROOT, "data", "databases", name + ".xlsx"))
        assert np.isclose(got[0], d_eff, rtol=1e-12, atol=0), (name, got[0], d_eff)
        assert np.allclose(got[1], eigenvalues, rtol=1e-12, atol=1e-15), (name, got[1])
        assert np.allclose(got[2], ratios, rtol=1e-12, atol=1e-15), (name, got[2])


def test_the_2d_projection_as_before():
    rng = np.random.default_rng(7)
    z, y = rng.normal(size=(20, 5)), rng.normal(size=(6, 5))
    pca = PCA(n_components=2).fit(z)
    assert pca.n_features_in_ == 5
    assert np.allclose(pca.transform(y), PROJECTED, rtol=0, atol=1e-12), pca.transform(y)
