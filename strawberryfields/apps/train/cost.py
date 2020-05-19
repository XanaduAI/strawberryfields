# Copyright 2020 Xanadu Quantum Technologies Inc.

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
TODO
"""
from typing import Callable

import numpy as np


class Stochastic:
    """TODO"""

    # TODO update type hints
    def __init__(self, h: Callable, vgbs):
        self.h = h
        self.vgbs = vgbs

    def evaluate(self, weights: np.ndarray) -> float:
        """TODO"""
        return

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """TODO"""
        return

    def __call__(self, weights: np.ndarray) -> float:
        return self.evaluate(weights)
