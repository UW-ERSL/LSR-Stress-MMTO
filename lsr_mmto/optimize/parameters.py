"""MMTO-specific extension of TOParams.

Only the fields the MMTO pipeline reads beyond the base TOParams live here:
  - MaterialFilterRadius : separate filter radius for the latent/material block
"""
from lsr_mmto.pyto_ad import TOParams  # github base TOParams, via the path-setup layer
from dataclasses import dataclass


@dataclass(slots=True)
class MMTOParams(TOParams):
    MaterialFilterRadius: float | None = None
