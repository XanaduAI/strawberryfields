r"""
Tools for training variational GBS devices.

.. currentmodule:: strawberryfields.apps.train

A GBS device can be treated as a variational quantum device with trainable squeezing,
beamsplitter and rotation gates. This module provides the functionality for training a GBS device
using the approach outlined in `this <https://arxiv.org/abs/2004.04770>`__ paper
:cite:`banchi2020training`.

Embedding trainable parameters in GBS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Training algorithms for GBS distributions rely on the :math:`WAW` parametrization, where :math:`W`
is a diagonal matrix of weights and :math:`A` is a symmetric matrix. Trainable parameters are
embedded into the GBS distribution by expressing the weights as functions of the parameters.

This module contains methods to implement such embeddings. It also provides derivatives
of the weights with respect to the trainable parameters. There are two main classes, each
corresponding to a different embedding. The :class:`Exp` class is a simple embedding where the
weights are exponentials of the trainable parameters. The :class:`ExpFeatures` class is a more
general embedding that makes use of user-defined feature vectors, which potentially provide more
flexibility in training strategies."

.. autosummary::
    :toctree: api

    Exp
    ExpFeatures

.. autosummary::
    :toctree: api

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
