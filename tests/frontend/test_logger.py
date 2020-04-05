# Copyright 2010 Pallets

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

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
r"""Unit tests for the logging mechanism used in Strawberry Fields.

The implementation of these unit tests is based on the solution used for testing logging in
the Flask web application framework:
https://github.com/pallets/flask/blob/master/tests/test_logging.py
"""

import logging

import pytest

import strawberryfields.api.job as job
import strawberryfields.api.connection as connection
import strawberryfields.engine as engine

from strawberryfields.logger import has_level_handler, default_handler, create_logger

modules_contain_logging = [job, connection, engine]


@pytest.mark.parametrize("module", modules_contain_logging)
class TestLogger:
    """Tests for the functions that are used to create a logger"""

    @pytest.fixture(autouse=True)
    def reset_logging(self, pytestconfig):
        """Reset the logging specific configurations such as handlers or levels as
        well as manage pytest's LoggingPlugin."""
        root_handlers = logging.root.handlers[:]
        logging.root.handlers = []
        root_level = logging.root.level

        logging_plugin = pytestconfig.pluginmanager.unregister(name="logging-plugin")

        yield

        logging.root.handlers[:] = root_handlers
        logging.root.setLevel(root_level)

        if logging_plugin:
            pytestconfig.pluginmanager.register(logging_plugin, "logging-plugin")

    def test_has_level_handler(self, module):
        """Tests the has_level_handler function"""
        logger = logging.getLogger(module.__name__)
        assert not has_level_handler(logger)

        handler = logging.StreamHandler()
        logging.root.addHandler(handler)
        assert has_level_handler(logger)

        logger.propagate = False
        assert not has_level_handler(logger)
        logger.propagate = True

        handler.setLevel(logging.ERROR)
        assert not has_level_handler(logger)

    def test_create_logger(self, module):
        """Tests the create_logger function"""
        logger = create_logger(module.__name__)
        assert logger.level == logging.DEBUG
        assert has_level_handler(logger)
        assert logger.handlers[0] == default_handler

class TestLoggerIntegration:
    """Tests that the SF logger integrates well with user defined logging configurations."""

    def test_custom_configuration_without_sf_logger(self, tmpdir, caplog):
        """Tests that if there was no SF logger created, custom logging
        configurations work as expected."""

        level = logging.DEBUG
        with caplog.at_level(level):

            test_file = tmpdir.join("test_file")
            logging.basicConfig(filename=test_file, level=level)
            logging.debug("A log entry.")

        assert "A log entry." in caplog.text

    @pytest.mark.parametrize("module", modules_contain_logging)
    def test_custom_configuration_after_sf_logger(self, module, tmpdir, caplog):
        """Tests that even if an SF logger was created, custom logging
        configurations work as expected."""

        level = logging.DEBUG
        with caplog.at_level(level):

            sf_logger = create_logger(module.__name__)
            test_file = tmpdir.join("test_file")
            logging.basicConfig(filename=test_file, level=level)
            logging.debug("A log entry.")

        assert "A log entry." in caplog.text

    @pytest.mark.parametrize("module", modules_contain_logging)
    def test_custom_logger_before_sf_logger_with_higher_level(self, module, tmpdir, caplog):
        """Tests that a custom logger created before an SF logger will define
        the level for logging as expected."""

        level = logging.DEBUG
        with caplog.at_level(level):

            test_file = tmpdir.join("test_file")
            logging.basicConfig(filename=test_file, level=level)
            logger = logging.getLogger(module.__name__)

            # Create a logger with a higher level
            # than the user defined one the value for
            # WARNING is higher than for DEBUG
            sf_logger = create_logger(module.__name__, level=logging.WARNING)
            logging.debug("A log entry.")

        assert "A log entry." in caplog.text
