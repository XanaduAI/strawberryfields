# Copyright 2019-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data module providing pre-calculated datasets from GBS simulations.

GBS samples
^^^^^^^^^^^
The following datasets of GBS samples are available:

.. currentmodule:: strawberryfields.apps.data

.. autosummary::
    :toctree: api

        Planted
        TaceAs
        PHat
        Mutag0
        Mutag1
        Mutag2
        Mutag3
        Formic
        Water
        Pyrrole

.. seealso::

    :doc:`/introduction/data`

The :mod:`~.sample` submodule contains the base :class:`~.GraphDataset`,
:class:`~.MoleculeDataset`, and :class:`~.SampleDataset` classes from which these datasets inherit.

.. autosummary::
    :toctree: api

        sample

GBS feature vectors
^^^^^^^^^^^^^^^^^^^

For use with the :mod:`~.similarity` module, the following pre-calculated feature vectors of graph
datasets are provided:

.. autosummary::
    :toctree: api

        MUTAG
        QM9Exact
        QM9MC

The :mod:`~.feature` submodule contains the base :class:`~.FeatureDataset` class from which
these datasets inherit.

.. autosummary::
    :toctree: api

        feature
"""

import strawberryfields.apps.data.sample
import strawberryfields.apps.data.feature
from strawberryfields.apps.data.sample import (
    Planted,
    TaceAs,
    PHat,
    Mutag0,
    Mutag1,
    Mutag2,
    Mutag3,
    Formic,
    Water,
    Pyrrole,
)
from strawberryfields.apps.data.feature import (
    MUTAG,
    QM9Exact,
    QM9MC,
)
