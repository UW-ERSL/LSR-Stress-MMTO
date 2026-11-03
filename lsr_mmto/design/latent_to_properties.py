"""
latent_to_properties.py
=======================
Each element's latent point z to its material properties, through the material network's
decoder, inside the torch tree: decode(enc, z, ...) gives E, nu and every decoded property.
"""
import torch

import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path (and its float64 default)


def decode(enc, z, use_constant_nu, nu_const):
    """E, nu per element, plus every decoded property, from the VAE decoder, inside the tree."""
    props = enc.getMaterialProperties(enc.vaeNet.decoder(z))
    E = props["Youngs_Modulus"]                 # column names are registry names (databases.ATTRIBUTES)
    nu = (torch.full_like(E, float(nu_const)) if (use_constant_nu or "Poissons_Ratio" not in props)
          else props["Poissons_Ratio"])
    return E, nu, props
