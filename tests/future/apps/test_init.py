# Copyright 2019 Xanadu Quantum Technologies Inc.
r"""
Unit tests for glassonion.__init__
"""
# pylint: disable=protected-access
import glassonion.__init__
import glassonion._version


def test_version_return_type():
    """Tests if ``glassonion.__init__.version`` returns a string"""
    assert isinstance(glassonion.__init__.version(), str)


def test_version_number():
    """Tests if ``glassonion.__init__.version`` returns the correct version"""
    assert glassonion.__init__.version() == glassonion._version.__version__
