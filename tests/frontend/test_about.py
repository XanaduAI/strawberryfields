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
Unit tests for top-level Strawberry Fields functions.
"""
import pytest
import re
import strawberryfields as sf


pytestmark = pytest.mark.frontend


def test_about(capfd):
    """sf.about works."""
    sf.about()
    out, err = capfd.readouterr()
    # substantial output (actual length varies)
    assert len(err) == 0
    assert len(out) > 300

    assert "Strawberry Fields version" in out
    sf_version_match = re.search(r"Strawberry Fields version:\s+([\S]+)\n", out).group(1)
    assert sf_version_match == sf.version()

    assert "Numpy version" in out
    assert "Scipy version" in out
    assert "The Walrus version" in out
    assert "Blackbird version" in out



def test_cite(capfd):
    """sf.cite works."""
    sf.cite()
    out, err = capfd.readouterr()
    # correct output
    assert len(err) == 0
    assert len(out) == 431
