r"""
Tools for training variational GBS devices.

.. currentmodule:: strawberryfields.apps.train

.. autosummary::
    :toctree: api

    embed
    Stochastic
    VGBS
    param
    KL
    cost
"""
from strawberryfields.apps.train.cost import KL, Stochastic
from strawberryfields.apps.train.embed import Exp, ExpFeatures
from strawberryfields.apps.train.vgbs import (
    VGBS,
    A_to_cov,
    prob_click,
    prob_photon_sample,
    rescale_adjacency,
)
