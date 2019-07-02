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
r"""Unit tests for Channel classes in ops.py"""
import pytest

pytestmark = pytest.mark.frontend

import numpy as np

import strawberryfields as sf

from strawberryfields import ops
from strawberryfields.program_utils import MergeFailure
from strawberryfields import utils
from strawberryfields.parameters import Parameter

# make test deterministic
np.random.seed(42)
a = np.random.random()
b = np.random.random()
c = np.random.random()


class TestChannelBasics:
    """Test the basic properties of channels"""

    def test_loss_merging(self, tol):
        """test the merging of two Loss channels (with default values
        for optional parameters)"""
        G = ops.LossChannel(a)
        merged = G.merge(ops.LossChannel(b))
        assert np.allclose(merged.p[0].x, a * b, atol=tol, rtol=0)

    def test_loss_merging_identity(self, tol):
        """test the merging of two Loss channels such that
        the resulting loss channel is simply the identity"""
        G = ops.LossChannel(a)
        merged = G.merge(ops.LossChannel(1.0 / a))
        assert merged is None

    def test_thermalloss_merging_same_nbar(self, tol):
        """test the merging of two Loss channels with same nbar"""
        G = ops.ThermalLossChannel(a, c)
        merged = G.merge(ops.ThermalLossChannel(b, c))
        assert np.allclose(merged.p[0].x, a * b, atol=tol, rtol=0)

    def test_thermalloss_merging_different_nbar(self, tol):
        """test the merging of two Loss channels with same nbar raises exception"""
        G = ops.ThermalLossChannel(a, 2 * c)
        with pytest.raises(MergeFailure):
            merged = G.merge(ops.ThermalLossChannel(b, c))

    def test_thermalloss_merging_different_nbar(self, tol):
        """test the merging of Loss and ThermalLoss raises exception"""
        G = ops.ThermalLossChannel(a, 2 * c)
        with pytest.raises(MergeFailure):
            merged = G.merge(ops.LossChannel(b))
