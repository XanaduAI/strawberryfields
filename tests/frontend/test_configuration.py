# Copyright 2019-2020 Xanadu Quantum Technologies Inc.

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

from strawberryfields import configuration as conf

pytestmark = pytest.mark.frontend
logging.getLogger().setLevel(1)

authentication_token = "071cdcce-9241-4965-93af-4a4dbc739135"

TEST_FILE = """\
[api]
# Options for the Strawberry Fields Cloud API
authentication_token = "071cdcce-9241-4965-93af-4a4dbc739135"
hostname = "localhost"
use_ssl = true
port = 443
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
        "port": 443,
        "debug": False,
    }
}

OTHER_EXPECTED_CONFIG = {
    "api": {
        "authentication_token": "SomeAuth",
        "hostname": "SomeHost",
        "use_ssl": False,
        "port": 56,
        "debug": True,
    }
}

environment_variables = [
                    "SF_API_AUTHENTICATION_TOKEN",
                    "SF_API_HOSTNAME",
                    "SF_API_USE_SSL",
                    "SF_API_DEBUG",
                    "SF_API_PORT"
                    ]

def tear_down_all_env_var_defs():
    """Making sure that no environment variables are defined after."""
    for key in environment_variables:
        if key in os.environ:
            del os.environ[key]
            assert key not in os.environ

class TestLoadConfig:
    """Tests for the load_config function."""

    def test_not_found_warning(self, caplog):
        """Test that a warning is raised if no configuration file found."""

        conf.load_config(filename='NotAFileName')
        assert "No Strawberry Fields configuration file found." in caplog.text

    def test_keywords_take_precedence_over_everything(self, monkeypatch, tmpdir):
        """Test that the keyword arguments passed to load_config take
        precedence over data in environment variables or data in a
        configuration file."""

        filename = tmpdir.join("config.toml")

        with open(filename, "w") as f:
            f.write(TEST_FILE)

        os.environ["SF_API_AUTHENTICATION_TOKEN"] = "NotOurAuth"
        os.environ["SF_API_HOSTNAME"] = "NotOurHost"
        os.environ["SF_API_USE_SSL"] = "True"
        os.environ["SF_API_DEBUG"] = "False"
        os.environ["SF_API_PORT"] = "42"

        with monkeypatch.context() as m:
            m.setattr(os, "getcwd", lambda: tmpdir)
            configuration = conf.load_config(authentication_token="SomeAuth",
                                            hostname="SomeHost",
                                            use_ssl=False,
                                            debug=True,
                                            port=56
                                            )

        assert configuration == OTHER_EXPECTED_CONFIG

    def test_environment_variables_take_precedence_over_conf_file(self, monkeypatch, tmpdir):
        """Test that the data in environment variables precedence over data in
        a configuration file."""

        filename = tmpdir.join("config.toml")

        with open(filename, "w") as f:
            f.write(TEST_FILE)

        os.environ["SF_API_AUTHENTICATION_TOKEN"] = "SomeAuth"
        os.environ["SF_API_HOSTNAME"] = "SomeHost"
        os.environ["SF_API_USE_SSL"] = "False"
        os.environ["SF_API_DEBUG"] = "True"
        os.environ["SF_API_PORT"] = "56"

        with monkeypatch.context() as m:
            m.setattr(os, "getcwd", lambda: tmpdir)
            configuration = conf.load_config()

        assert configuration == OTHER_EXPECTED_CONFIG

        tear_down_all_env_var_defs()

    def test_conf_file_loads_well(self, monkeypatch, tmpdir):
        """Test that the data in environment variables precedence over data in
        a configuration file."""

        filename = tmpdir.join("config.toml")

        with open(filename, "w") as f:
            f.write(TEST_FILE)

        with monkeypatch.context() as m:
            m.setattr(os, "getcwd", lambda: tmpdir)
            configuration = conf.load_config()

        assert configuration == EXPECTED_CONFIG

class TestCreateConfigObject:
    """Test the creation of a configuration object"""

    def test_empty_config_object(self):
        """Test that an empty configuration object can be created."""
        config = conf.create_config_object(authentication_token="",
                                  hostname="",
                                  use_ssl="",
                                  debug="",
                                  port="")

        assert all(value=="" for value in config["api"].values())

    def test_config_object_with_authentication_token(self):
        """Test that passing only the authentication token creates the expected
        configuration object."""
        assert conf.create_config_object(authentication_token="071cdcce-9241-4965-93af-4a4dbc739135") == EXPECTED_CONFIG

    def test_config_object_every_keyword_argument(self):
        """Test that passing every keyword argument creates the expected
        configuration object."""
        assert conf.create_config_object(authentication_token="SomeAuth",
                                        hostname="SomeHost",
                                        use_ssl=False,
                                        debug=True,
                                        port=56) == OTHER_EXPECTED_CONFIG
class TestLookForConfigInFile:
    """Tests for the look_for_config_in_file function."""

    def test_loading_current_directory(self, tmpdir, monkeypatch):
        """Test that the default configuration file is loaded from the current
        directory, if found."""
        filename = "config.toml"

        with monkeypatch.context() as m:
            m.setattr(os, "getcwd", lambda: tmpdir)
            m.setattr(conf, "load_config_file", lambda filepath: filepath)
            config_file = conf.look_for_config_in_file(filename=filename)

        assert config_file == tmpdir.join(filename)

    def test_loading_env_variable(self, tmpdir, monkeypatch):
        """Test that the correct configuration file is found using the correct
        environment variable.

        This is a test case for when there is no configuration file in the
        current directory."""

        filename = "config.toml"

        def raise_wrapper(ex):
            raise ex

        with monkeypatch.context() as m:
            m.setattr(os, "getcwd", lambda: "NoConfigFileHere")
            m.setattr(os.environ, "get", lambda x, y: tmpdir if x=="SF_CONF" else "NoConfigFileHere")
            m.setattr(conf, "load_config_file", lambda filepath: raise_wrapper(FileNotFoundError()) if "NoConfigFileHere" in filepath else filepath)

            # Need to mock the module specific function
            # m.setattr(conf, "user_config_dir", lambda *args: "NotTheFileName")
            # os.environ["SF_CONF"] = lambda: FileNotFoundError
            config_file = conf.look_for_config_in_file(filename=filename)
        assert config_file == tmpdir.join("config.toml")

    def test_loading_user_config_dir(self, tmpdir, monkeypatch):
        """Test that the correct configuration file is found using the correct
        argument to the user_config_dir function.

        This is a test case for when there is no configuration file:
        -in the current directory or
        -in the directory contained in the corresponding environment
        variable."""
        filename = "config.toml"

        def raise_wrapper(ex):
            raise ex

        with monkeypatch.context() as m:
            m.setattr(os, "getcwd", lambda: "NoConfigFileHere")
            m.setattr(os.environ, "get", lambda *args: "NoConfigFileHere")
            m.setattr(conf, "user_config_dir", lambda x, *args: tmpdir if x=="strawberryfields" else "NoConfigFileHere")
            m.setattr(conf, "load_config_file", lambda filepath: raise_wrapper(FileNotFoundError()) if "NoConfigFileHere" in filepath else filepath)

            config_file = conf.look_for_config_in_file(filename=filename)
        assert config_file == tmpdir.join("config.toml")

    def test_no_config_file_found_returns_none(self, tmpdir, monkeypatch):
        """Test that the the look_for_config_in_file returns None if the
        configuration file is nowhere to be found.

        This is a test case for when there is no configuration file:
        -in the current directory or
        -in the directory contained in the corresponding environment
        variable
        -in the user_config_dir directory of Strawberry Fields."""
        filename = "config.toml"

        def raise_wrapper(ex):
            raise ex

        with monkeypatch.context() as m:
            m.setattr(os, "getcwd", lambda: "NoConfigFileHere")
            m.setattr(os.environ, "get", lambda *args: "NoConfigFileHere")
            m.setattr(conf, "user_config_dir", lambda *args: "NoConfigFileHere")
            m.setattr(conf, "load_config_file", lambda filepath: raise_wrapper(FileNotFoundError()) if "NoConfigFileHere" in filepath else filepath)

            config_file = conf.look_for_config_in_file(filename=filename)

        assert config_file is None

    def test_load_config_file(self, tmpdir, monkeypatch):
        """Tests that configuration is loaded correctly from a TOML file."""
        filename = tmpdir.join("config.toml")

        with open(filename, "w") as f:
            f.write(TEST_FILE)

        config_file = conf.load_config_file(filepath=filename)

        assert config_file == EXPECTED_CONFIG

class TestUpdateWithOtherConfig:
    """Tests for the update_with_other_config function."""

    def test_update_entire_config(self):
        """Tests that the entire configuration object is updated."""

        config = conf.create_config_object()
        assert config["api"]["authentication_token"] == ""

        conf.update_with_other_config(config, EXPECTED_CONFIG)
        assert config == EXPECTED_CONFIG

    ONLY_AUTH_CONFIG = {
                "api": {
                        "authentication_token": "PlaceHolder",
                                                            }
                    }

    ONLY_HOST_CONFIG = {
                        "api": {
                                    "hostname": "PlaceHolder",
                                }
                    }

    ONLY_SSL_CONFIG = {
                "api": {
                        "use_ssl": "PlaceHolder",
                                                            }
    }

    ONLY_DEBUG_CONFIG = {
                "api": {
                        "debug": "PlaceHolder",
                                                            }
    }

    ONLY_PORT_CONFIG = {
            "api": {"port": "PlaceHolder"}
    }

    @pytest.mark.parametrize("specific_key, config_to_update_with", [("authentication_token",ONLY_AUTH_CONFIG),
                                                        ("hostname",ONLY_HOST_CONFIG),
                                                        ("use_ssl",ONLY_SSL_CONFIG),
                                                        ("debug",ONLY_DEBUG_CONFIG),
                                                        ("port",ONLY_PORT_CONFIG)])
    def test_update_only_one_item_in_section(self, specific_key, config_to_update_with):
        """Tests that only one item is updated in the configuration object is updated."""
        config = conf.create_config_object()
        assert config["api"][specific_key] != "PlaceHolder"

        conf.update_with_other_config(config, config_to_update_with)
        assert config["api"][specific_key] == "PlaceHolder"
        assert all(v != "PlaceHolder" for k, v in config["api"].items() if k != specific_key)


value_mapping = [
                ("SF_API_AUTHENTICATION_TOKEN","SomeAuth"),
                ("SF_API_HOSTNAME","SomeHost"),
                ("SF_API_USE_SSL","False"),
                ("SF_API_PORT","56"),
                ("SF_API_DEBUG","True")
                ]

parsed_values_mapping = {
                "SF_API_AUTHENTICATION_TOKEN": "SomeAuth",
                "SF_API_HOSTNAME": "SomeHost",
                "SF_API_USE_SSL": False,
                "SF_API_PORT": 56,
                "SF_API_DEBUG": True,
                        }

class TestUpdateFromEnvironmentalVariables:
    """Tests for the update_from_environment_variables function."""

    def test_all_environment_variables_defined(self):
        """Tests that the configuration object is updated correctly when all
        the environment variables are defined."""

        for key, value in value_mapping:
            os.environ[key] = value

        config = conf.create_config_object()
        for v, parsed_value in zip(config["api"].values(), parsed_values_mapping.values()):
            assert v != parsed_value

        conf.update_from_environment_variables(config)
        for v, parsed_value in zip(config["api"].values(), parsed_values_mapping.values()):
            assert v == parsed_value

        tear_down_all_env_var_defs()

    environment_variables_with_keys_and_values = [
                    ("SF_API_AUTHENTICATION_TOKEN","authentication_token","SomeAuth"),
                    ("SF_API_HOSTNAME","hostname","SomeHost"),
                    ("SF_API_USE_SSL","use_ssl","False"),
                    ("SF_API_PORT","port", "56"),
                    ("SF_API_DEBUG","debug","True")
                    ]

    @pytest.mark.parametrize("env_var, key, value", environment_variables_with_keys_and_values)
    def test_one_environment_variable_defined(self, env_var, key, value):
        """Tests that the configuration object is updated correctly when only
        one environment variable is defined."""

        tear_down_all_env_var_defs()
        os.environ[env_var] = value

        config = conf.create_config_object()
        for v, parsed_value in zip(config["api"].values(), parsed_values_mapping.values()):
            assert v != parsed_value

        conf.update_from_environment_variables(config)
        assert config["api"][key] == parsed_values_mapping[env_var]

        for v, (key, parsed_value) in zip(config["api"].values(), parsed_values_mapping.items()):
            if key != env_var:
                assert v != parsed_value

        # Tear-down
        del os.environ[env_var]
        assert env_var not in os.environ

    def test_parse_environment_variable_boolean(self, monkeypatch):
        """Tests that boolean values can be parsed correctly from environment
        variables."""
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

        assert conf.parse_environment_variable("not_a_boolean","something_else") == "something_else"

    def test_parse_environment_variable_integer(self, monkeypatch):
        """Tests that integer values can be parsed correctly from environment
        variables."""

        monkeypatch.setattr(conf, "INTEGER_KEYS", ("some_integer",))
        assert conf.parse_environment_variable("some_integer", "123") == 123
        assert conf.parse_environment_variable("not_an_integer","something_else") == "something_else"

