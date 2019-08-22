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
r"""Unit tests for TensorFlow 1.3 version checking"""
from importlib import reload
import sys

import pytest

import strawberryfields as sf


try:
    import tensorflow
except (ImportError, ModuleNotFoundError):
    tf_available = False
    import mock
    tensorflow = mock.Mock(__version__="1.12.2")
else:
    tf_available = True


class TestBackendImport:
    """Test importing the backend directly"""

    def test_incorrect_tf_version(self, monkeypatch):
        """Test that an exception is raised if the version
        of TensorFlow installed is not version 1.3"""
        with monkeypatch.context() as m:
            # force Python check to pass
            m.setattr("sys.version_info", (3, 6, 3))
            m.setattr(tensorflow, "__version__", "1.12.2")

            with pytest.raises(ImportError, match="version 1.3 of TensorFlow is required"):
                reload(sf.backends.tfbackend)

    def test_incorrect_python_version(self, monkeypatch):
        """Test that an exception is raised if the version
        of Python installed is > 3.6"""
        with monkeypatch.context() as m:
            m.setattr("sys.version_info", (3, 8, 1))
            m.setattr(tensorflow, "__version__", "1.12.2")

            with pytest.raises(ImportError, match="you will need to install Python 3.6"):
                reload(sf.backends.tfbackend)

    @pytest.mark.skipif(tf_available, reason="Test only works if TF not installed")
    def test_tensorflow_not_installed(self, monkeypatch):
        """Test that an exception is raised if TensorFlow is not installed"""
        with monkeypatch.context() as m:
            # force Python check to pass
            m.setattr("sys.version_info", (3, 6, 3))

            with pytest.raises(ImportError, match="version 1.3 of TensorFlow is required"):
                reload(sf.backends.tfbackend)


@pytest.mark.frontend
class TestFrontendImport:
    """Test importing via the frontend"""

    def test_incorrect_tf_version(self, monkeypatch):
        """Test that an exception is raised if the version
        of TensorFlow installed is not version 1.3"""
        with monkeypatch.context() as m:
            # force Python check to pass
            m.setattr("sys.version_info", (3, 6, 3))
            m.setattr(tensorflow, "__version__", "1.12.2")

            with pytest.raises(ImportError, match="version 1.3 of TensorFlow is required"):
                reload(sf.backends.tfbackend)
                sf.LocalEngine('tf')

    def test_incorrect_python_version(self, monkeypatch):
        """Test that an exception is raised if the version
        of Python installed is > 3.6"""
        with monkeypatch.context() as m:
            m.setattr("sys.version_info", (3, 8, 1))
            m.setattr(tensorflow, "__version__", "1.12.2")

            with pytest.raises(ImportError, match="you will need to install Python 3.6"):
                reload(sf.backends.tfbackend)
                sf.LocalEngine('tf')

    @pytest.mark.skipif(tf_available, reason="Test only works if TF not installed")
    def test_tensorflow_not_installed(self, monkeypatch):
        """Test that an exception is raised if TensorFlow is not installed"""
        with monkeypatch.context() as m:
            # force Python check to pass
            m.setattr("sys.version_info", (3, 6, 3))

            with pytest.raises(ImportError, match="version 1.3 of TensorFlow is required"):
                reload(sf.backends.tfbackend)
                sf.LocalEngine('tf')
