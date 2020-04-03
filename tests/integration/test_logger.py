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
r"""Integration tests for the logging mechanism used in Strawberry Fields."""

import logging
import pytest

import strawberryfields.api.job as job
import strawberryfields.api.connection as connection
import strawberryfields.engine as engine

from strawberryfields.logger import has_level_handler, default_handler, create_logger

modules_contain_logging = [job, connection, engine]


class TestLoggerIntegration:
    """Tests that the SF logger integrates well with user defined logging configurations."""

    def test_custom_configuration_without_sf_logger(self, tmpdir, caplog):
        """Tests that if there was no SF logger created, custom logging
        configurations work as expected."""

        level = logging.DEBUG
        with caplog.at_level(level):

            test_file = tmpdir.join("test_file")
            logging.basicConfig(filename=test_file, level=level)
            logging.debug('A log entry.')

        assert 'A log entry.' in caplog.text

    @pytest.mark.parametrize("module", modules_contain_logging)
    def test_custom_configuration_after_sf_logger(self, module, tmpdir, caplog):
        """Tests that even if an SF logger was created, custom logging
        configurations work as expected."""

        level = logging.DEBUG
        with caplog.at_level(level):

            sf_logger = create_logger(module.__name__)
            test_file = tmpdir.join("test_file")
            logging.basicConfig(filename=test_file, level=level)
            logging.debug('A log entry.')

        assert 'A log entry.' in caplog.text

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
            logging.debug('A log entry.')

        assert 'A log entry.' in caplog.text

