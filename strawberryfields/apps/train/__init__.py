r"""
Tools for training variational GBS devices.

.. currentmodule:: strawberryfields.apps.train

.. autosummary::
    :toctree: api

    embed
    VGBS
    param
    KL
    cost
"""
from strawberryfields.apps.train.param import VGBS
from strawberryfields.apps.train.cost import KL
import strawberryfields.apps.train.cost
import strawberryfields.apps.train.param
import strawberryfields.apps.train.embed
