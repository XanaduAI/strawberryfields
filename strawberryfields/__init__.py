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
The Strawberry Fields codebase includes a number of complementary components.
These can be separated into frontend components, applications layer,
and backend components (all found within the :mod:`strawberryfields.backends` submodule).

.. image:: ../_static/sfcomponents.svg
    :align: center
    :width: 90%
    :target: javascript:void(0);
"""
from . import apps
from ._version import __version__
from .cli import ping
from .configuration import store_account, active_configs, reset_config, delete_config
from .engine import Engine, LocalEngine, RemoteEngine
from .io import load, save
from .parameters import par_funcs as math
from .program import Program
from .tdm import TDMProgram
from .plot import plot_wigner, plot_fock, plot_quad
from . import tdm

__all__ = [
    "Engine",
    "RemoteEngine",
    "Program",
    "TDMProgram",
    "version",
    "save",
    "load",
    "about",
    "cite",
    "ping",
    "store_account",
    "active_configs",
    "reset_config",
    "delete_config",
]


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
    """Strawberry Fields information.

    Prints the installed version numbers for SF and its dependencies,
    and some system info. Please include this information in bug reports.

    **Example:**

    .. code-block:: pycon

        >>> sf.about()
        Strawberry Fields: a Python library for continuous-variable quantum circuits.
        Copyright 2018-2020 Xanadu Quantum Technologies Inc.

        Python version:            3.6.10
        Platform info:             Linux-5.8.18-1-MANJARO-x86_64-with-arch-Manjaro-Linux
        Installation path:         /home/strawberryfields/
        Strawberry Fields version: 0.17.0
        Numpy version:             1.19.5
        Scipy version:             1.4.1
        SymPy version:             1.5.1
        NetworkX version:          2.4
        The Walrus version:        0.14.0
        Blackbird version:         0.3.0
        TensorFlow version:        2.4.0
    """
    # pylint: disable=import-outside-toplevel
    import sys
    import platform
    import os
    import numpy
    import scipy
    import sympy
    import networkx
    import thewalrus
    import blackbird

    # a QuTiP-style infobox
    print("\nStrawberry Fields: a Python library for continuous-variable quantum circuits.")
    print("Copyright 2018-2020 Xanadu Quantum Technologies Inc.\n")

    print("Python version:            {}.{}.{}".format(*sys.version_info[0:3]))
    print("Platform info:             {}".format(platform.platform()))
    print("Installation path:         {}".format(os.path.dirname(__file__)))
    print("Strawberry Fields version: {}".format(__version__))
    print("Numpy version:             {}".format(numpy.__version__))
    print("Scipy version:             {}".format(scipy.__version__))
    print("SymPy version:             {}".format(sympy.__version__))
    print("NetworkX version:          {}".format(networkx.__version__))
    print("The Walrus version:        {}".format(thewalrus.__version__))
    print("Blackbird version:         {}".format(blackbird.__version__))

    try:
        import tensorflow

        tf_version = tensorflow.__version__
    except ImportError:
        tf_version = None

    print("TensorFlow version:        {}".format(tf_version))


def cite():
    """Prints the BibTeX citation for Strawberry Fields.

    **Example:**

    .. code-block:: pycon

        >>> sf.cite()
        @article{strawberryfields,
            title = {{S}trawberry {F}ields: A Software Platform for Photonic Quantum Computing},
            author = {Killoran, Nathan and Izaac, Josh and Quesada, Nicol{'{a}}s and Bergholm, Ville and Amy, Matthew and Weedbrook, Christian},
            journal = {Quantum},
            volume = {3},
            pages = {129},
            year = {2019},
            doi = {10.22331/q-2019-03-11-129},
            archivePrefix = {arXiv},
            eprint = {1804.03159},
        }
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
