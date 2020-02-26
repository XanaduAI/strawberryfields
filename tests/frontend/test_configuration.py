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
    }
}

OTHER_EXPECTED_CONFIG = {
    "api": {
        "authentication_token": "SomeAuth",
        "hostname": "SomeHost",
        "use_ssl": False,
        "port": 56,
    }
}

environment_variables = [
                    "SF_API_AUTHENTICATION_TOKEN",
                    "SF_API_HOSTNAME",
                    "SF_API_USE_SSL",
                    "SF_API_DEBUG",
                    "SF_API_PORT"
                    ]

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

        with monkeypatch.context() as m:
            m.setenv("SF_API_AUTHENTICATION_TOKEN", "NotOurAuth")
            m.setenv("SF_API_HOSTNAME", "NotOurHost")
            m.setenv("SF_API_USE_SSL", "True")
            m.setenv("SF_API_DEBUG", "False")
            m.setenv("SF_API_PORT", "42")

            m.setattr(os, "getcwd", lambda: tmpdir)
            configuration = conf.load_config(authentication_token="SomeAuth",
                                            hostname="SomeHost",
                                            use_ssl=False,
                                            port=56
                                            )

        assert configuration == OTHER_EXPECTED_CONFIG

    def test_environment_variables_take_precedence_over_conf_file(self, monkeypatch, tmpdir):
        """Test that the data in environment variables take precedence over data in
        a configuration file."""

        filename = tmpdir.join("config.toml")

        with open(filename, "w") as f:
            f.write(TEST_FILE)

        with monkeypatch.context() as m:
            m.setattr(os, "getcwd", lambda: tmpdir)

            m.setenv("SF_API_AUTHENTICATION_TOKEN", "SomeAuth")
            m.setenv("SF_API_HOSTNAME", "SomeHost")
            m.setenv("SF_API_USE_SSL", "False")
            m.setenv("SF_API_DEBUG", "True")
            m.setenv("SF_API_PORT", "56")

            configuration = conf.load_config()

        assert configuration == OTHER_EXPECTED_CONFIG

    def test_conf_file_loads_well(self, monkeypatch, tmpdir):
        """Test that the load_config function loads a configuration from a TOML
        file correctly."""

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
        config = conf.create_config(authentication_token="",
                                  hostname="",
                                  use_ssl="",
                                  port="")

        assert all(value=="" for value in config["api"].values())

    def test_config_object_with_authentication_token(self):
        """Test that passing only the authentication token creates the expected
        configuration object."""
        assert conf.create_config(authentication_token="071cdcce-9241-4965-93af-4a4dbc739135") == EXPECTED_CONFIG

    def test_config_object_every_keyword_argument(self):
        """Test that passing every keyword argument creates the expected
        configuration object."""
        assert conf.create_config(authentication_token="SomeAuth",
                                        hostname="SomeHost",
                                        use_ssl=False,
                                        port=56) == OTHER_EXPECTED_CONFIG
class TestGetConfigFilepath:
    """Tests for the get_config_filepath function."""

    def test_current_directory(self, tmpdir, monkeypatch):
        """Test that the default configuration file is loaded from the current
        directory, if found."""
        filename = "config.toml"

        path_to_write_file = tmpdir.join(filename)

        with open(path_to_write_file, "w") as f:
            f.write(TEST_FILE)

        with monkeypatch.context() as m:
            m.setattr(os, "getcwd", lambda: tmpdir)
            config_filepath = conf.get_config_filepath(filename=filename)

        assert config_filepath == tmpdir.join(filename)

    def test_env_variable(self, tmpdir, monkeypatch):
        """Test that the correct configuration file is found using the correct
        environment variable (SF_CONF).

        This is a test case for when there is no configuration file in the
        current directory."""

        filename = "config.toml"

        path_to_write_file = tmpdir.join(filename)

        with open(path_to_write_file, "w") as f:
            f.write(TEST_FILE)

        def raise_wrapper(ex):
            raise ex

        with monkeypatch.context() as m:
            m.setattr(os, "getcwd", lambda: "NoConfigFileHere")
            m.setenv("SF_CONF", tmpdir)
            m.setattr(conf, "user_config_dir", lambda *args: "NotTheFileName")

            config_filepath = conf.get_config_filepath(filename=filename)

        assert config_filepath == tmpdir.join("config.toml")

    def test_user_config_dir(self, tmpdir, monkeypatch):
        """Test that the correct configuration file is found using the correct
        argument to the user_config_dir function.

        This is a test case for when there is no configuration file:
        -in the current directory or
        -in the directory contained in the corresponding environment
        variable."""
        filename = "config.toml"

        path_to_write_file = tmpdir.join(filename)

        with open(path_to_write_file, "w") as f:
            f.write(TEST_FILE)

        def raise_wrapper(ex):
            raise ex

        with monkeypatch.context() as m:
            m.setattr(os, "getcwd", lambda: "NoConfigFileHere")
            m.setenv("SF_CONF", "NoConfigFileHere")
            m.setattr(conf, "user_config_dir", lambda x, *args: tmpdir if x=="strawberryfields" else "NoConfigFileHere")

            config_filepath = conf.get_config_filepath(filename=filename)

        assert config_filepath == tmpdir.join("config.toml")

    def test_no_config_file_found_returns_none(self, tmpdir, monkeypatch):
        """Test that the get_config_filepath returns None if the
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
            m.setenv("SF_CONF", "NoConfigFileHere")
            m.setattr(conf, "user_config_dir", lambda *args: "NoConfigFileHere")

            config_filepath = conf.get_config_filepath(filename=filename)

        assert config_filepath is None

class TestLoadConfigFile:
    """Tests the load_config_file function."""

    def test_load_config_file(self, tmpdir, monkeypatch):
        """Tests that configuration is loaded correctly from a TOML file."""
        filename = tmpdir.join("config.toml")

        with open(filename, "w") as f:
            f.write(TEST_FILE)

        loaded_config = conf.load_config_file(filepath=filename)

        assert loaded_config == EXPECTED_CONFIG

    def test_loading_absolute_path(self, tmpdir, monkeypatch):
        """Test that the default configuration file can be loaded
        via an absolute path."""
        filename = os.path.abspath(tmpdir.join("config.toml"))


        with open(filename, "w") as f:
            f.write(TEST_FILE)

        with monkeypatch.context() as m:
            m.setenv("SF_CONF", "")
            loaded_config = conf.load_config_file(filepath=filename)

        assert loaded_config == EXPECTED_CONFIG

class TestKeepValidOptions:

    def test_only_invalid_options(self):
        section_config_with_invalid_options = {'NotValid1': 1,
                                               'NotValid2': 2,
                                               'NotValid3': 3
                                              }
        assert conf.keep_valid_options(section_config_with_invalid_options) == {}

    def test_valid_and_invalid_options(self):
        section_config_with_invalid_options = { 'authentication_token': 'MyToken',
                                                'NotValid1': 1,
                                               'NotValid2': 2,
                                               'NotValid3': 3
                                              }
        assert conf.keep_valid_options(section_config_with_invalid_options) == {'authentication_token': 'MyToken'}

    def test_only_valid_options(self):
        section_config_only_valid = {
                                    "authentication_token": "071cdcce-9241-4965-93af-4a4dbc739135",
                                    "hostname": "localhost",
                                    "use_ssl": True,
                                    "port": 443,
                                    }
        assert conf.keep_valid_options(section_config_only_valid) == EXPECTED_CONFIG["api"]

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

    def test_all_environment_variables_defined(self, monkeypatch):
        """Tests that the configuration object is updated correctly when all
        the environment variables are defined."""

        with monkeypatch.context() as m:
            for env_var, value in value_mapping:
                m.setenv(env_var, value)

            config = conf.create_config()
            for v, parsed_value in zip(config["api"].values(), parsed_values_mapping.values()):
                assert v != parsed_value

            conf.update_from_environment_variables(config)
            for v, parsed_value in zip(config["api"].values(), parsed_values_mapping.values()):
                assert v == parsed_value


    environment_variables_with_keys_and_values = [
                    ("SF_API_AUTHENTICATION_TOKEN","authentication_token","SomeAuth"),
                    ("SF_API_HOSTNAME","hostname","SomeHost"),
                    ("SF_API_USE_SSL","use_ssl","False"),
                    ("SF_API_PORT","port", "56"),
                    ]

    @pytest.mark.parametrize("env_var, key, value", environment_variables_with_keys_and_values)
    def test_one_environment_variable_defined(self, env_var, key, value, monkeypatch):
        """Tests that the configuration object is updated correctly when only
        one environment variable is defined."""

        with monkeypatch.context() as m:
            m.setenv(env_var, value)

            config = conf.create_config()
            for v, parsed_value in zip(config["api"].values(), parsed_values_mapping.values()):
                assert v != parsed_value

            conf.update_from_environment_variables(config)
            assert config["api"][key] == parsed_values_mapping[env_var]

            for v, (key, parsed_value) in zip(config["api"].values(), parsed_values_mapping.items()):
                if key != env_var:
                    assert v != parsed_value

    def test_parse_environment_variable_boolean(self, monkeypatch):
        """Tests that boolean values can be parsed correctly from environment
        variables."""
        monkeypatch.setattr(conf, "DEFAULT_CONFIG_SPEC", {"api": {"some_boolean": (bool, True)}})
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

    def test_parse_environment_variable_integer(self, monkeypatch):
        """Tests that integer values can be parsed correctly from environment
        variables."""

        monkeypatch.setattr(conf, "DEFAULT_CONFIG_SPEC", {"api": {"some_integer": (int, 123)}})
        assert conf.parse_environment_variable("some_integer", "123") == 123

DEFAULT_KWARGS = {
        "api": {
            "authentication_token": "071cdcce-9241-4965-93af-4a4dbc739135",
            "hostname": "localhost",
            "use_ssl": True,
            "port": 443,
                }
        }

class TestStoreAccount:
    """Tests for the store_account function."""

    def test_config_created_locally(self):
        """Tests that a configuration file was created in the current
        directory."""

        test_filename = "test_config.toml"



        call_history = []
        m.setattr(os, "getcwd", lambda: tmpdir)
        m.setattr(conf, "save_config_to_file", lambda a, b: call_history.append((a, b)))
        conf.store_account(authentication_token, filename=test_filename, create_locally=True, **DEFAULT_KWARGS)

        assert call_history[0] == DEFAULT_CONFIG
        assert call_history[0] == tmpdir.join(test_filename)

class TestSaveConfigToFile:
    """Tests for the store_account function."""

    def test_save(self, tmpdir):
        """Test saving a configuration file."""
        filename = str(tmpdir.join("test_config.toml"))	

        config = EXPECTED_CONFIG

        # make a change	
        config["api"]["hostname"] = "https://6.4.2.4"	
        conf.save_config_to_file(filename)

        result = toml.load(filename)	
        assert config == result
