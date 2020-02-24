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
debug = false
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
        "debug": False,
        "port": 443,
    }
}


class TestCreteConfigObject:
    def test_empty_config_object(self):
        config = conf.create_config_object(authentication_token="",
                                  hostname="",
                                  use_ssl="",
                                  debug="",
                                  port="")

        assert all(value=="" for value in config["api"].values())
    def test_config_object_with_authentication_token(self):
        assert conf.create_config_object(authentication_token="071cdcce-9241-4965-93af-4a4dbc739135") == EXPECTED_CONFIG

class TestConfiguration:
    """Tests for the configuration class"""

    def test_load_config_file(self, tmpdir, monkeypatch):
        filename = tmpdir.join("config.toml")

        with open(filename, "w") as f:
            f.write(TEST_FILE)

        config_file = conf.load_config_file(filepath=filename)

        assert config_file == EXPECTED_CONFIG

class TestLoadConfig:

    def test_not_found_warning(self, caplog):
        """Test that a warning is raised if no configuration file found."""

        conf.load_config(filename='NotAFileName')
        assert "No Strawberry Fields configuration file found." in caplog.text

    def test_check_call_order(self, monkeypatch):

        def mock_look_for_config_in_file(*args, **kwargs):
            call_history.append(2)
            return "NotNone"

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(conf, "create_config_object", lambda *args: call_history.append(1))
            m.setattr(conf, "look_for_config_in_file", mock_look_for_config_in_file)
            m.setattr(conf, "update_with_other_config", lambda *args, **kwargs: call_history.append(3))
            m.setattr(conf, "update_from_environment_variables", lambda *args: call_history.append(4))
            conf.load_config()
        assert call_history == [1,2,3,4]

class TestLookForConfigInFile:

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

    class TestUpdateWithOtherConfig:

        def test_update_entire_config(self):
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
            config = conf.create_config_object()
            assert config["api"][specific_key] != "PlaceHolder"

            conf.update_with_other_config(config, config_to_update_with)
            assert config["api"][specific_key] == "PlaceHolder"
            assert all(v != "PlaceHolder" for k, v in config["api"].items() if k != specific_key)

environment_variables = [
                    "SF_API_AUTHENTICATION_TOKEN",
                    "SF_API_HOSTNAME",
                    "SF_API_USE_SSL",
                    "SF_API_DEBUG",
                    "SF_API_PORT"
                    ]

class TestUpdateFromEnvironmentalVariables:

    def test_all_environment_variables_defined(self):

        for key in environment_variables:
            os.environ[key] = "PlaceHolder"

        config = conf.create_config_object()
        assert not any(v == "PlaceHolder" for k, v in config["api"].items())

        conf.update_from_environment_variables(config)
        assert all(v == "PlaceHolder" for k, v in config["api"].items())
    def test_one_environment_variable_defined(self, specific_env_var, specific_key):
        # Tear-down
        for key in environment_variables:
            del os.environ[key]
            assert key not in os.environ

    environment_variables_with_keys = [
                    ("SF_API_AUTHENTICATION_TOKEN","authentication_token"),
                    ("SF_API_HOSTNAME","hostname"),
                    ("SF_API_USE_SSL","use_ssl"),
                    ("SF_API_DEBUG","debug"),
                    ("SF_API_PORT","port")
                    ]

    @pytest.mark.parametrize("specific_env_var, specific_key", environment_variables_with_keys)
    def test_one_environment_variable_defined(self, specific_env_var, specific_key):
        # Making sure that no environment variable was defined previously
        for key in environment_variables:
            if key in os.environ:
                del os.environ[key]
                assert key not in os.environ

        os.environ[specific_env_var] = "PlaceHolder"

        config = conf.create_config_object()
        assert not any(v == "PlaceHolder" for k, v in config["api"].items())

        conf.update_from_environment_variables(config)
        assert config["api"][specific_key] == "PlaceHolder"

        assert all(v != "PlaceHolder" for k, v in config["api"].items() if k != specific_key)

        # Tear-down
        del os.environ[specific_env_var]
        assert specific_env_var not in os.environ

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

        assert conf.parse_environment_variable("not_a_boolean","something_else") == "something_else"

