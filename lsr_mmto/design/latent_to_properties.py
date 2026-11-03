"""
latent_to_properties.py
=======================
Each element's latent point z to its material properties, through the material network's
decoder, inside the torch tree: decode(enc, z) gives every property the database has, by its
column name (databases.ATTRIBUTES). Which of them a physics uses, and how, is the physics' own
(its analyse).
"""
import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path (and its float64 default)


def decode(enc, z):
    """Every decoded property per element, {column name: torch tensor}, from the VAE decoder, inside the tree."""
    return enc.getMaterialProperties(enc.vaeNet.decoder(z))
