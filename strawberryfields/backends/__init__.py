# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module loads the required backend classes"""

from .base import BaseBackend, BaseFock, BaseGaussian, ModeMap
from .tfbackend import TFBackend
from .gaussianbackend import GaussianBackend
from .fockbackend import FockBackend

__all__ = ['BaseBackend', 'BaseFock', 'BaseGaussian', 'FockBackend', 'GaussianBackend', 'TFBackend']

supported_backends = {"base": BaseBackend,
                      "tf": TFBackend,
                      "gaussian": GaussianBackend,
                      "fock": FockBackend}

def load_backend(name):
    """Loads the specified backend by mapping a string
    to the backend type, via the ``supported_backends``
    dictionary. Note that this function is used by the
    frontend only, and should not be user-facing.
    """
    if name in supported_backends:
        backend = supported_backends[name]()
        return backend
    else:
        raise ValueError("Backend '{}' is not supported.".format(name))
