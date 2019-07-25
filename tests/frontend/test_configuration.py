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
"""Unit tests for the configuration module"""
import os
import logging
import pytest

import toml

from unittest.mock import MagicMock

from strawberryfields import configuration as conf

pytestmark = pytest.mark.frontend
logging.getLogger().setLevel(1)


TEST_FILE = """\
[api]
# Options for the Strawberry Fields Cloud API
authentication_token = "071cdcce-9241-4965-93af-4a4dbc739135"
hostname = "localhost"
use_ssl = true
debug = false
"""

TEST_FILE_ONE_VALUE = """\
[api]
# Options for the Strawberry Fields Cloud API
authentication_token = "071cdcce-9241-4965-93af-4a4dbc739135"
"""

EXPECTED_CONFIG = {
    "api": {
        "authentication_token": "071cdcce-9241-4965-93af-4a4dbc739135",
        "hostname": "localhost",
        "use_ssl": True,
        "debug": False,
    }
}


class TestConfiguration:
    """Tests for the configuration class"""

    def test_loading_current_directory(self, tmpdir, monkeypatch):
        """Test that the default configuration file can be loaded
        from the current directory."""
        filename = tmpdir.join("config.toml")

        with open(filename, "w") as f:
            f.write(TEST_FILE)

        with monkeypatch.context() as m:
            m.setattr(os, "getcwd", lambda: str(tmpdir))
            os.environ["SF_CONF"] = ""
            config = conf.Configuration()

        assert config._config == EXPECTED_CONFIG
        assert config.path == filename

    def test_loading_env_variable(self, tmpdir):
        """Test that the default configuration file can be loaded
        via an environment variable."""
        filename = tmpdir.join("config.toml")

        with open(filename, "w") as f:
            f.write(TEST_FILE)

        os.environ["SF_CONF"] = str(tmpdir)
        config = conf.Configuration()

        assert config._config == EXPECTED_CONFIG
        assert config.path == filename

    def test_loading_absolute_path(self, tmpdir, monkeypatch):
        """Test that the default configuration file can be loaded
        via an absolute path."""
        filename = os.path.abspath(tmpdir.join("config.toml"))

        with open(filename, "w") as f:
            f.write(TEST_FILE)

        os.environ["SF_CONF"] = ""
        config = conf.Configuration(name=str(filename))

        assert config._config == EXPECTED_CONFIG
        assert config.path == filename

    def test_not_found_warning(self, caplog):
        """Test that a warning is raised if no configuration file found."""

        conf.Configuration(name="noconfig")
        assert "No Strawberry Fields configuration file found." in caplog.text

    def test_save(self, tmpdir):
        """Test saving a configuration file."""
        filename = str(tmpdir.join("test_config.toml"))
        config = conf.Configuration()

        # make a change
        config._config["api"]["hostname"] = "https://6.4.2.4"
        config.save(filename)

        result = toml.load(filename)
        assert config._config == result

    def test_attribute_loading(self):
        """Test attributes automatically get the correct section key"""
        config = conf.Configuration()
        assert config.api == config._config["api"]

    def test_failed_attribute_loading(self):
        """Test an exception is raised if key does not exist"""
        config = conf.Configuration()
        with pytest.raises(
            conf.ConfigurationError, match="Unknown Strawberry Fields configuration section"
        ):
            config.test

    def test_env_vars_take_precedence(self, tmpdir):
        """Test that if a configuration file and an environment
        variable is set, that the environment variable takes
        precedence."""
        filename = tmpdir.join("config.toml")

        with open(filename, "w") as f:
            f.write(TEST_FILE)

        host = "https://6.4.2.4"

        os.environ["SF_API_HOSTNAME"] = host
        config = conf.Configuration(str(filename))

        assert config.api["hostname"] == host

    def test_parse_environment_variable(self, monkeypatch):
        monkeypatch.setattr(conf, "BOOLEAN_KEYS", ("some_boolean",))
        assert conf.parse_environment_variable("some_boolean", "true") is True
        assert conf.parse_environment_variable("some_boolean", "True") is True
        assert conf.parse_environment_variable("some_boolean", "TRUE") is True
        assert conf.parse_environment_variable("some_boolean", "1") is True
        assert conf.parse_environment_variable("some_boolean", 1) is True

        assert conf.parse_environment_variable("some_boolean", "false") is False
        assert conf.parse_environment_variable("some_boolean", "False") is False
        assert conf.parse_environment_variable("some_boolean", "FALSE") is False
        assert conf.parse_environment_variable("some_boolean", "0") is False
        assert conf.parse_environment_variable("some_boolean", 0) is False

        something_else = MagicMock()
        assert conf.parse_environment_variable("not_a_boolean", something_else) == something_else

    def test_update_config_with_limited_config_file(self, tmpdir, monkeypatch):
        """
        This test asserts that the given a config file that only provides a single
        value, the rest of the configuration values are filled in using defaults.
        """
        filename = tmpdir.join("config.toml")

        with open(filename, "w") as f:
            f.write(TEST_FILE_ONE_VALUE)

        config = conf.Configuration(str(filename))
        assert config.api["hostname"] == conf.DEFAULT_CONFIG["api"]["hostname"]
        assert config.api["use_ssl"] == conf.DEFAULT_CONFIG["api"]["use_ssl"]
        assert config.api["authentication_token"] == "071cdcce-9241-4965-93af-4a4dbc739135"
