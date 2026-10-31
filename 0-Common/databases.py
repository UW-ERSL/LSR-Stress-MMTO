"""
databases.py
============
The material databases shipped with the repository, and the VAE settings each was
trained with, so the network shipped beside each database (.nt) can be loaded.

A database is an .xlsx file in DATABASE_DIR: row 1 the attribute names, row 2 their
units, then one row per material.
"""
import os
from dataclasses import dataclass

DATABASE_DIR = os.path.join("1-PureStructural", "MaterialDataPureStructural")


@dataclass(slots=True)
class VAEParams:
    """VAE settings for a database that has none of its own."""
    klFactor: float = 5e-6
    learningRate: float = 2e-6
    numEpochs: int = 200000                 # max epochs
    vae_hiddenDim: int = 500
    latentDim: int = 2
    maxAttributeErrorPercent: float = 1     # stops training when every attribute is this close


# the settings each shipped network was trained with
DATABASE_VAE = {
    "BM1_3Mats_2Attr":                dict(latentDim=2, learningRate=5e-4, vae_hiddenDim=50,
                                           maxAttributeErrorPercent=0.1, klFactor=1e-4, numEpochs=30000),
    "6Materials_AF_3Attr":            dict(latentDim=2, learningRate=1e-4, vae_hiddenDim=50,
                                           maxAttributeErrorPercent=0.1, klFactor=1e-4, numEpochs=100000),
    "Beam_8Mats_3Attr_v4_spectrum2D": dict(latentDim=2, learningRate=5e-4, vae_hiddenDim=50,
                                           maxAttributeErrorPercent=0.1, klFactor=1e-4, numEpochs=100000),
    "CustomAlloys20_3D":              dict(latentDim=3, learningRate=5e-4, vae_hiddenDim=50,
                                           maxAttributeErrorPercent=0.1, klFactor=1e-4, numEpochs=100000),
}
