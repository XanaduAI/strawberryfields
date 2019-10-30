# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Vibronic Spectra
================

**Module name:** :mod:`strawberryfields.gbs.vibronic`

.. currentmodule:: strawberryfields.gbs.vibronic

This module contains functions for computing molecular vibronic spectra using GBS.
An accompanying tutorial can be found :ref:`here <>`.

Vibronic Spectroscopy
---------------------

Summary
-------

.. autosummary::
    gbs_params
    energies

Code details
^^^^^^^^^^^^
"""

import numpy as np


def energies(samples: list, wp: np.ndarray) -> list:
    r"""Computes the energy of each GBS sample in units of :math:`\text{cm}^{-1}`.

    **Example usage:**

    >>> samples = np.array([[1, 1, 0], [1, 0, 2]])
    >>> wp = np.array([700.0, 600.0, 500.0])
    >>> energies(samples, wp)
    [1300.0, 1700.0]

    Args:
        samples (array): GBS samples
        wp (array): normal mode frequencies in units of :math:`\text{cm}^{-1}`

    Returns:
        E (list[int]): list of GBS sample energies in units of :math:`\text{cm}^{-1}`
    """
    return [np.dot(s, wp) for s in samples]
