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
"""
.. _code:

Overview
==================

.. image:: ../_static/sfcomponents.svg
    :align: center
    :width: 90%
    :target: javascript:void(0);

|

The Strawberry Fields codebase includes a number of complementary components.
These can be separated into frontend components
and backend components (all found within the :file:`strawberryfields.backends` submodule).

Software components
-------------------

**Frontend:**

* Quantum programs: :mod:`strawberryfields.program`
* Quantum execution and compilation engine: :mod:`strawberryfields.engine`
* Quantum operations: :mod:`strawberryfields.ops`
* Input/output functions: :mod:`strawberryfields.io`
* Circuit specifications: :mod:`strawberryfields.circuitspecs`
* Decompositions: :mod:`strawberryfields.decompositions`
* Utilities: :mod:`strawberryfields.utils`
* Circuit drawer: :mod:`strawberryfields.circuitdrawer`

**Backend:**

* Backend API: :mod:`strawberryfields.backends.base`
* Quantum states: :mod:`strawberryfields.backends.states`
* Fock simulator backend: :mod:`strawberryfields.backends.fockbackend`
* Gaussian simulator backend: :mod:`strawberryfields.backends.gaussianbackend`
* Tensorflow simulator backend: :mod:`strawberryfields.backends.tfbackend`

Top-level functions
-------------------

.. currentmodule: strawberryfields

.. autosummary::
   convert
   about
   cite
   version

Code details
~~~~~~~~~~~~
"""
from .engine import (Engine, LocalEngine)
from .io import save, load
from .program_utils import _convert as convert
from .program import Program
from ._version import __version__

__all__ = ["Engine", "LocalEngine", "Program", "convert", "version", "save", "load", "about", "cite"]


#: float: numerical value of hbar for the frontend (in the implicit units of position * momentum)
hbar = 2


def version():
    r"""
    Version number of Strawberry Fields.

    Returns:
      str: package version number
    """
    return __version__


def about():
    """About box for Strawberry Fields.

    Prints the installed version numbers for SF and its dependencies,
    and some system info. Please include this information in bug reports.
    """
    import sys
    import platform
    import os
    import numpy
    import scipy
    import thewalrus
    import blackbird

    # a QuTiP-style infobox
    print('\nStrawberry Fields: a Python library for continuous-variable quantum circuits.')
    print('Copyright 2018-2019 Xanadu Quantum Technologies Inc.\n')

    print('Python version:            {}.{}.{}'.format(*sys.version_info[0:3]))
    print('Platform info:             {}'.format(platform.platform()))
    print('Installation path:         {}'.format(os.path.dirname(__file__)))
    print('Strawberry Fields version: {}'.format(__version__))
    print('Numpy version:             {}'.format(numpy.__version__))
    print('Scipy version:             {}'.format(scipy.__version__))
    print('The Walrus version:           {}'.format(thewalrus.__version__))
    print('Blackbird version:         {}'.format(blackbird.__version__))

    try:
        import tensorflow
        tf_version = tensorflow.__version__
    except ImportError:
        tf_version = None

    print('TensorFlow version:        {}'.format(tf_version))


def cite():
    """Prints the BibTeX citation for Strawberry Fields.

    BibTex code for reference :cite:`strawberryfields`.
    """
    citation = """@article{strawberryfields,
    title = {{S}trawberry {F}ields: A Software Platform for Photonic Quantum Computing},
    author = {Killoran, Nathan and Izaac, Josh and Quesada, Nicol{\'{a}}s and Bergholm, Ville and Amy, Matthew and Weedbrook, Christian},
    journal = {Quantum},
    volume = {3},
    pages = {129},
    year = {2019},
    doi = {10.22331/q-2019-03-11-129},
    archivePrefix = {arXiv},
    eprint = {1804.03159},
}"""
    print(citation)
