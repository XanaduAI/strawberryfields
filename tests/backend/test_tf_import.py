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
import pytest


def test_incorrect_tf_version(monkeypatch):
    """Test that an exception is raised if the version
    of TensorFlow installed is not version 1.3"""
    with monkeypatch.context() as m:
        # force Python check to pass
        m.setattr("sys.version_info", (3, 6, 3))
        m.setattr("tensorflow.__version__", "1.12.2")

        with pytest.raises(ImportError, message="version 1.3 of TensorFlow is required"):
            from strawberryfields.backends.tfbackend import TFBackend


def test_incorrect_python_version(monkeypatch):
    """Test that an exception is raised if the version
    of Python installed is > 3.6"""
    with monkeypatch.context() as m:
        m.setattr("sys.version_info", (3, 8, 1))

        with pytest.raises(ImportError, message="you will need to install Python 3.6"):
            from strawberryfields.backends.tfbackend import TFBackend


def test_tensorflow_not_installed(monkeypatch):
    """Test that an exception is raised if TensorFlow is not installed"""
    with monkeypatch.context() as m:
        # force Python check to pass
        m.setattr("sys.version_info", (3, 6, 3))
        m.delitem("sys.modules", "tensorflow", raising=False)

        with pytest.raises(ImportError, message="version 1.3 of TensorFlow is required"):
            from strawberryfields.backends.tfbackend import TFBackend
