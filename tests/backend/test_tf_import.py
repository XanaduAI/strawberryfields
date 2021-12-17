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
r"""Unit tests for TensorFlow 2.x version checking"""
import sys
from unittest.mock import MagicMock

import pytest

import strawberryfields as sf

try:
    import tensorflow

except ImportError:
    tf_available = False

else:
    tf_available = True


@pytest.mark.frontend
class TestFrontendImport:
    """Test importing via the frontend"""

    def test_incorrect_tf_version(self, monkeypatch):
        """Test that an exception is raised if the version
        of TensorFlow installed is not version 2.x"""
        with monkeypatch.context() as m:
            # Force the Python check to pass.
            m.setattr(sys, "version_info", (3, 6, 3))

            # Unload the TF backend to ensure sf.LocalEngine() will run __init__.py.
            m.delitem(sys.modules, "strawberryfields.backends.tfbackend", raising=False)
            # Set the TF version in case the existing version is valid.
            m.setitem(sys.modules, "tensorflow", MagicMock(__version__="1.2.3"))

            with pytest.raises(ImportError, match="version 2.x of TensorFlow is required"):
                sf.LocalEngine("tf")

    @pytest.mark.skipif(tf_available, reason="Test only works if TF not installed")
    def test_tensorflow_not_installed(self, monkeypatch):
        """Test that an exception is raised if TensorFlow is not installed"""
        with monkeypatch.context() as m:
            # Force the Python check to pass.
            m.setattr(sys, "version_info", (3, 6, 3))

            with pytest.raises(ImportError, match="version 2.x of TensorFlow is required"):
                sf.LocalEngine("tf")
