r"""
Tools for training variational GBS devices.

.. currentmodule:: strawberryfields.apps.train

.. autosummary::
    :toctree: api

    embed
    Stochastic
    VGBS
    param
"""
from strawberryfields.apps.train.param import VGBS
import strawberryfields.apps.train.param
import strawberryfields.apps.train.embed
from strawberryfields.apps.train.cost import Stochastic
