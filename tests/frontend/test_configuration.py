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
hostname = "platform.strawberryfields.ai"
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
        "hostname": "platform.strawberryfields.ai",
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
    "SF_API_PORT",
]


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_not_found_warning(self, caplog):
        """Test that a warning is raised if no configuration file found."""

        conf.load_config(filename="NotAFileName")
        assert "No Strawberry Fields configuration file found." in caplog.text

    def test_keywords_take_precedence_over_everything(self, monkeypatch, tmpdir):
        """Test that the keyword arguments passed to load_config take
        precedence over data in environment variables or data in a
        configuration file."""
        with open(tmpdir.join("test_config.toml"), "w") as f:
            f.write(TEST_FILE)

        with monkeypatch.context() as m:
            m.setenv("SF_API_AUTHENTICATION_TOKEN", "NotOurAuth")
            m.setenv("SF_API_HOSTNAME", "NotOurHost")
            m.setenv("SF_API_USE_SSL", "True")
            m.setenv("SF_API_DEBUG", "False")
            m.setenv("SF_API_PORT", "42")

            m.setattr(os, "getcwd", lambda: tmpdir)
            configuration = conf.load_config(
                authentication_token="SomeAuth", hostname="SomeHost", use_ssl=False, port=56
            )

        assert configuration == OTHER_EXPECTED_CONFIG

    def test_environment_variables_take_precedence_over_conf_file(self, monkeypatch, tmpdir):
        """Test that the data in environment variables take precedence over data in
        a configuration file."""
        with open(tmpdir.join("test_config.toml"), "w") as f:
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

    def test_get_api_section_safely_error(self, monkeypatch, tmpdir, caplog):
        """Test that the get_api_section_safely function raises an error and
        logs correctly if there is no api section in the configuration file."""
        filename = tmpdir.join("config.toml")

        empty_file = ""

        with open(filename, "w") as f:
            f.write(empty_file)

        with monkeypatch.context() as m:
            with pytest.raises(conf.ConfigurationError, match=""):
                m.setattr(os, "getcwd", lambda: tmpdir)
                configuration = conf.load_config()

        assert 'does not contain an "api" section' in caplog.text

    def test_directories_to_check_with_sf_conf(self, monkeypatch, tmpdir):
        """Test that the directories_to_check function returns three
        directories if the SF_CONF variables is defined."""
        with monkeypatch.context() as m:
            m.setattr(os, "getcwd", lambda: "First")
            m.setenv("SF_CONF", "Second")
            m.setattr(conf, "user_config_dir", lambda *args: "Third")
            directories = conf.directories_to_check()

            assert ["First", "Second", "Third"] == directories

    def test_directories_to_check_without_sf_conf(self, monkeypatch, tmpdir):
        """Test that the directories_to_check function returns two
        directories if the SF_CONF variables is not defined."""
        with monkeypatch.context() as m:
            m.setattr(os, "getcwd", lambda: "First")
            m.delenv("SF_CONF", raising=False)
            m.setattr(conf, "user_config_dir", lambda *args: "Second")
            directories = conf.directories_to_check()

            assert ["First", "Second"] == directories


test_file_name = "test_file.toml"
default_file_name = "config.toml"
test_kwargs = [{}, {"filename": test_file_name}]


class TestActiveConfigs:
    """Test the active_configs function and its auxiliary functions."""

    @pytest.mark.parametrize("kwargs", test_kwargs)
    def test_get_available_config_paths_mock_directories(self, monkeypatch, kwargs):
        """Test that the get_available_config_paths correctly uses the output of the
        directories_to_check function."""
        test_directory = "test_directory_path"
        test_file_name = kwargs["filename"] if kwargs else default_file_name

        with monkeypatch.context() as m:
            m.setattr(conf, "directories_to_check", lambda: [test_directory])
            m.setattr(os.path, "exists", lambda *args: True)
            active_configs = conf.get_available_config_paths(**kwargs)
            assert active_configs == [os.path.join(test_directory, test_file_name)]

    @pytest.mark.parametrize("kwargs", test_kwargs)
    def test_get_available_config_paths_integration(self, monkeypatch, tmpdir, kwargs):
        """Tests that the get_available_config_paths function integrates well with
        parts of the directories_to_check function."""
        test_file_name = kwargs["filename"] if kwargs else default_file_name

        path1 = tmpdir.mkdir("sub1")
        path2 = tmpdir.mkdir("sub2")
        path3 = tmpdir.mkdir("sub3")

        file1 = path1.join(test_file_name)
        file2 = path2.join(test_file_name)
        file3 = path3.join(test_file_name)

        temporary_files = [file1, file2, file3]

        with monkeypatch.context() as m:
            m.setattr(os, "getcwd", lambda: path1)
            m.setenv("SF_CONF", str(path2))
            m.setattr(conf, "user_config_dir", lambda *args: path3)
            m.setattr(os.path, "exists", lambda arg: arg in temporary_files)
            active_configs = conf.get_available_config_paths(**kwargs)
            assert active_configs == [file1, file2, file3]

    def test_print_active_configs_single_config(self, capsys, monkeypatch):
        """Checks that the correct message is outputted when a single
        configuration file was found."""
        active_configs = ["first_path"]
        temp_dirs = ["first_path", "second_path", "third_path"]
        with monkeypatch.context() as m:
            m.setattr(conf, "directories_to_check", lambda: temp_dirs)
            m.setattr(conf, "get_available_config_paths", lambda *args: active_configs)
            conf.active_configs(test_file_name)

            captured = capsys.readouterr()

            general_message_1 = (
                "\nThe following Strawberry Fields configuration files were found "
                'with the name "{}":\n'.format(test_file_name)
            )
            single_config = "\n* " + active_configs[0] + " (active)\n"

            general_message_2 = "\nThe following directories were checked:\n\n"

            first_dir_msg = "* " + temp_dirs[0] + "\n"
            second_dir_msg = "* " + temp_dirs[1] + "\n"
            third_dir_msg = "* " + temp_dirs[2] + "\n"

            assert (
                captured.out
                == general_message_1
                + single_config
                + general_message_2
                + first_dir_msg
                + second_dir_msg
                + third_dir_msg
            )

    def test_print_active_configs_multiple_configs(self, capsys, monkeypatch):
        """Checks that the correct message is outputted for a single
        configuration file found."""
        active_configs = ["first_path", "second_path", "third_path"]
        temp_dirs = ["first_path", "second_path", "third_path"]
        with monkeypatch.context() as m:
            m.setattr(conf, "directories_to_check", lambda: temp_dirs)
            m.setattr(conf, "get_available_config_paths", lambda *args: active_configs)
            conf.active_configs(test_file_name)

            captured = capsys.readouterr()

            general_message_1 = (
                "\nThe following Strawberry Fields configuration files were found "
                'with the name "{}":\n'.format(test_file_name)
            )
            first_config = "\n* " + active_configs[0] + " (active)\n"
            second_config = "* " + active_configs[1] + "\n"
            third_config = "* " + active_configs[2] + "\n"

            general_message_2 = "\nThe following directories were checked:\n\n"

            first_dir_msg = "* " + temp_dirs[0] + "\n"
            second_dir_msg = "* " + temp_dirs[1] + "\n"
            third_dir_msg = "* " + temp_dirs[2] + "\n"

            assert (
                captured.out
                == general_message_1
                + first_config
                + second_config
                + third_config
                + general_message_2
                + first_dir_msg
                + second_dir_msg
                + third_dir_msg
            )

    def test_print_active_configs_no_configs(self, capsys, monkeypatch):
        """Checks that the correct message is outputted if no configuration
        files were found."""
        temp_dirs = ["first_path", "second_path", "third_path"]
        with monkeypatch.context() as m:
            m.setattr(conf, "directories_to_check", lambda: temp_dirs)
            m.setattr(conf, "get_available_config_paths", lambda *args: [])
            conf.active_configs(test_file_name)

            captured = capsys.readouterr()

            general_message_1 = (
                "\nNo Strawberry Fields configuration files were found with the "
                'name "{}".\n\n'.format(test_file_name)
            )

            general_message_2 = "\nThe following directories were checked:\n\n"

            first_dir_msg = "* " + temp_dirs[0] + "\n"
            second_dir_msg = "* " + temp_dirs[1] + "\n"
            third_dir_msg = "* " + temp_dirs[2] + "\n"

            assert (
                captured.out
                == general_message_1
                + general_message_2
                + first_dir_msg
                + second_dir_msg
                + third_dir_msg
            )


class TestCreateConfigObject:
    """Test the creation of a configuration object"""

    def test_empty_config_object(self):
        """Test that an empty configuration object can be created."""
        config = conf.create_config(authentication_token="", hostname="", use_ssl="", port="")

        assert all(value == "" for value in config["api"].values())

    def test_config_object_with_authentication_token(self):
        """Test that passing only the authentication token creates the expected
        configuration object."""
        assert (
            conf.create_config(authentication_token="071cdcce-9241-4965-93af-4a4dbc739135")
            == EXPECTED_CONFIG
        )

    def test_config_object_every_keyword_argument(self):
        """Test that passing every keyword argument creates the expected
        configuration object."""
        assert (
            conf.create_config(
                authentication_token="SomeAuth", hostname="SomeHost", use_ssl=False, port=56
            )
            == OTHER_EXPECTED_CONFIG
        )


class TestRemoveConfigFile:
    """Test the removal of configuration files"""

    def test_remove_default_config(self, monkeypatch, tmpdir):
        """Test removing the default config file; i.e. called without arguments"""
        filename = tmpdir.join("config.toml")
        empty_file = ""

        with open(filename, "w") as f:
            f.write(empty_file)

        with monkeypatch.context() as m:
            m.setattr(conf, "find_config_file", lambda *args: filename)

            assert os.path.exists(filename)

            conf.delete_config()

            assert not os.path.exists(filename)

    def test_remove_config_in_subdirectory(self, tmpdir):
        """Test removing a config file in a subdirectory"""
        subdir = tmpdir.join("subdir")
        filename = subdir.join("new_config.toml")
        empty_file = ""

        os.mkdir(subdir)
        with open(filename, "w") as f:
            f.write(empty_file)

        assert os.path.exists(filename)

        conf.delete_config(filename, directory=subdir)

        assert not os.path.exists(filename)

    def test_reset_config(self, monkeypatch, tmpdir):
        """Test resetting the configuration by removing current configuration files"""
        subdir = tmpdir.join("subdir")
        filename_1 = tmpdir.join("config.toml")
        filename_2 = subdir.join("config.toml")
        active_configs = [filename_1, filename_2]

        empty_file = ""

        with open(filename_1, "w") as f:
            f.write(empty_file)

        os.mkdir(subdir)
        with open(filename_2, "w") as f:
            f.write(empty_file)

        with monkeypatch.context() as m:
            m.setattr(conf, "get_available_config_paths", lambda *args: active_configs)

            assert os.path.exists(filename_1)
            assert os.path.exists(filename_2)

            conf.reset_config()

            assert not os.path.exists(filename_1)
            assert not os.path.exists(filename_2)


class TestGetConfigFilepath:
    """Tests for the find_config_file function."""

    def test_current_directory(self, tmpdir, monkeypatch):
        """Test that the default configuration file is loaded from the current
        directory, if found."""
        filename = "config.toml"

        path_to_write_file = tmpdir.join(filename)

        with open(path_to_write_file, "w") as f:
            f.write(TEST_FILE)

        with monkeypatch.context() as m:
            m.setattr(os, "getcwd", lambda: tmpdir)
            config_filepath = conf.find_config_file(filename=filename)

        assert config_filepath == tmpdir.join(filename)

    def test_env_variable(self, monkeypatch, tmpdir):
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
            m.setenv("SF_CONF", str(tmpdir))
            m.setattr(conf, "user_config_dir", lambda *args: "NotTheFileName")

            config_filepath = conf.find_config_file(filename="config.toml")

        assert config_filepath == tmpdir.join("config.toml")

    def test_user_config_dir(self, monkeypatch, tmpdir):
        """Test that the correct configuration file is found using the correct
        argument to the user_config_dir function.

        This is a test case for when there is no configuration file:
        -in the current directory or
        -in the directory contained in the corresponding environment
        variable."""
        with open(tmpdir.join("config.toml"), "w") as f:
            f.write(TEST_FILE)

        def raise_wrapper(ex):
            raise ex

        with monkeypatch.context() as m:
            m.setattr(os, "getcwd", lambda: "NoConfigFileHere")
            m.setenv("SF_CONF", "NoConfigFileHere")
            m.setattr(
                conf,
                "user_config_dir",
                lambda x, *args: tmpdir if x == "strawberryfields" else "NoConfigFileHere",
            )

            config_filepath = conf.find_config_file(filename="config.toml")

        assert config_filepath == tmpdir.join("config.toml")

    def test_no_config_file_found_returns_none(self, monkeypatch, tmpdir):
        """Test that the find_config_file returns None if the
        configuration file is nowhere to be found.

        This is a test case for when there is no configuration file:
        -in the current directory or
        -in the directory contained in the corresponding environment
        variable
        -in the user_config_dir directory of Strawberry Fields."""

        def raise_wrapper(ex):
            raise ex

        with monkeypatch.context() as m:
            m.setattr(os, "getcwd", lambda: "NoConfigFileHere")
            m.setenv("SF_CONF", "NoConfigFileHere")
            m.setattr(conf, "user_config_dir", lambda *args: "NoConfigFileHere")

            config_filepath = conf.find_config_file(filename="config.toml")

        assert config_filepath is None


class TestLoadConfigFile:
    """Tests the load_config_file function."""

    def test_load_config_file(self, monkeypatch, tmpdir):
        """Tests that configuration is loaded correctly from a TOML file."""
        filename = tmpdir.join("test_config.toml")

        with open(filename, "w") as f:
            f.write(TEST_FILE)

        loaded_config = conf.load_config_file(filepath=filename)

        assert loaded_config == EXPECTED_CONFIG

    def test_loading_absolute_path(self, monkeypatch, tmpdir):
        """Test that the default configuration file can be loaded
        via an absolute path."""
        filename = tmpdir.join("test_config.toml")

        with open(filename, "w") as f:
            f.write(TEST_FILE)

        with monkeypatch.context() as m:
            m.setenv("SF_CONF", "")
            loaded_config = conf.load_config_file(filepath=filename)

        assert loaded_config == EXPECTED_CONFIG


class TestKeepValidOptions:
    def test_only_invalid_options(self):
        section_config_with_invalid_options = {"NotValid1": 1, "NotValid2": 2, "NotValid3": 3}
        assert conf.keep_valid_options(section_config_with_invalid_options) == {}

    def test_valid_and_invalid_options(self):
        section_config_with_invalid_options = {
            "authentication_token": "MyToken",
            "NotValid1": 1,
            "NotValid2": 2,
            "NotValid3": 3,
        }
        assert conf.keep_valid_options(section_config_with_invalid_options) == {
            "authentication_token": "MyToken"
        }

    def test_only_valid_options(self):
        section_config_only_valid = {
            "authentication_token": "071cdcce-9241-4965-93af-4a4dbc739135",
            "hostname": "platform.strawberryfields.ai",
            "use_ssl": True,
            "port": 443,
        }
        assert conf.keep_valid_options(section_config_only_valid) == EXPECTED_CONFIG["api"]


value_mapping = [
    ("SF_API_AUTHENTICATION_TOKEN", "SomeAuth"),
    ("SF_API_HOSTNAME", "SomeHost"),
    ("SF_API_USE_SSL", "False"),
    ("SF_API_PORT", "56"),
    ("SF_API_DEBUG", "True"),
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
        ("SF_API_AUTHENTICATION_TOKEN", "authentication_token", "SomeAuth"),
        ("SF_API_HOSTNAME", "hostname", "SomeHost"),
        ("SF_API_USE_SSL", "use_ssl", "False"),
        ("SF_API_PORT", "port", "56"),
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

            for v, (key, parsed_value) in zip(
                config["api"].values(), parsed_values_mapping.items()
            ):
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


DEFAULT_KWARGS = {"hostname": "platform.strawberryfields.ai", "use_ssl": True, "port": 443}


class MockSaveConfigToFile:
    """A mock class used to contain the state left by the save_config_to_file
    function."""

    def __init__(self):
        self.config = None
        self.path = None

    def update(self, config, path):
        """Updates the instance attributes."""
        self.config = config
        self.path = path


def mock_create_config(authentication_token="", **kwargs):
    """A mock version of the create_config function adjusted to the
    store_account function.
    """
    return {"api": {"authentication_token": authentication_token, **kwargs}}


class TestStoreAccount:
    """Tests for the store_account function."""

    def test_config_created_locally(self, monkeypatch, tmpdir):
        """Tests that a configuration file was created in the current
        directory."""
        mock_save_config_file = MockSaveConfigToFile()

        assert mock_save_config_file.config is None
        assert mock_save_config_file.path is None

        with monkeypatch.context() as m:
            m.setattr(os, "getcwd", lambda: tmpdir)
            m.setattr(conf, "user_config_dir", lambda *args: "NotTheCorrectDir")
            m.setattr(conf, "create_config", mock_create_config)
            m.setattr(conf, "save_config_to_file", lambda a, b: mock_save_config_file.update(a, b))
            conf.store_account(
                authentication_token, filename="config.toml", location="local", **DEFAULT_KWARGS
            )

        assert mock_save_config_file.config == EXPECTED_CONFIG
        assert mock_save_config_file.path == tmpdir.join("config.toml")

    def test_global_config_created(self, monkeypatch, tmpdir):
        """Tests that a configuration file was created in the user
        configuration directory for Strawberry Fields."""
        mock_save_config_file = MockSaveConfigToFile()

        assert mock_save_config_file.config is None
        assert mock_save_config_file.path is None

        with monkeypatch.context() as m:
            m.setattr(os, "getcwd", lambda: "NotTheCorrectDir")
            m.setattr(conf, "user_config_dir", lambda *args: tmpdir)
            m.setattr(conf, "create_config", mock_create_config)
            m.setattr(conf, "save_config_to_file", lambda a, b: mock_save_config_file.update(a, b))
            conf.store_account(
                authentication_token,
                filename="config.toml",
                location="user_config",
                **DEFAULT_KWARGS
            )

        assert mock_save_config_file.config == EXPECTED_CONFIG
        assert mock_save_config_file.path == tmpdir.join("config.toml")

    def test_location_not_recognized_error(self, monkeypatch, tmpdir):
        """Tests that an error is raised if the configuration file is supposed
        to be created in an unrecognized directory."""

        with pytest.raises(conf.ConfigurationError, match="This location is not recognized."):
            conf.store_account(
                authentication_token,
                filename="config.toml",
                location="UNRECOGNIZED_LOCATION",
                **DEFAULT_KWARGS
            )

    def test_non_existing_directory_does_not_raise_file_not_found_error(self, monkeypatch, tmpdir):
        """Tests that an error is raised if the configuration file is supposed
        to be created in a non-existing directory when using user_config_dir
        and if os.makedirs does not create the directory."""

        with monkeypatch.context() as m:
            m.setattr(conf, "user_config_dir", lambda *args: tmpdir.join("new_dir"))
            conf.store_account(
                authentication_token,
                filename="config.toml",
                location="user_config",
                **DEFAULT_KWARGS
            )

    def test_non_existing_directory_without_makedirs_raises_error(self, monkeypatch, tmpdir):
        """Tests that an error is raised if the configuration file is supposed
        to be created in a non-existing directory when using user_config_dir
        and if os.makedirs does not create the directory."""

        with monkeypatch.context() as m:
            m.setattr(os, "makedirs", lambda a, **kwargs: None)
            m.setattr(conf, "user_config_dir", lambda *args: tmpdir.join("new_dir"))
            with pytest.raises(FileNotFoundError, match="No such file or directory"):
                conf.store_account(
                    authentication_token,
                    filename="config.toml",
                    location="user_config",
                    **DEFAULT_KWARGS
                )


class TestStoreAccountIntegration:
    """Integration tests for the store_account function.

    Mocking takes place only such that writing can be done in pytest's
    temporary directory.
    """

    def test_local(self, monkeypatch, tmpdir):
        """Tests that the functions integrate correctly when storing account
        locally."""

        with monkeypatch.context() as m:
            m.setattr(os, "getcwd", lambda: tmpdir)
            conf.store_account(
                authentication_token, filename="config.toml", location="local", **DEFAULT_KWARGS
            )

        filepath = tmpdir.join("config.toml")
        result = toml.load(filepath)
        assert result == EXPECTED_CONFIG

    def test_global(self, monkeypatch, tmpdir):
        """Tests that the functions integrate correctly when storing account
        globally."""

        with monkeypatch.context() as m:
            m.setattr(conf, "user_config_dir", lambda *args: tmpdir)
            conf.store_account(
                authentication_token,
                filename="config.toml",
                location="user_config",
                **DEFAULT_KWARGS
            )

        filepath = tmpdir.join("config.toml")
        result = toml.load(filepath)
        assert result == EXPECTED_CONFIG

    def test_directory_is_created(self, monkeypatch, tmpdir):

        recursive_dir = tmpdir.join(".new_dir")
        with monkeypatch.context() as m:
            m.setattr(conf, "user_config_dir", lambda *args: recursive_dir)
            conf.store_account(
                authentication_token,
                filename="config.toml",
                location="user_config",
                **DEFAULT_KWARGS
            )

        filepath = os.path.join(recursive_dir, "config.toml")
        result = toml.load(filepath)
        assert result == EXPECTED_CONFIG

    def test_nested_directory_is_created(self, monkeypatch, tmpdir):

        recursive_dir = tmpdir.join(".new_dir", "new_dir_again")
        with monkeypatch.context() as m:
            m.setattr(conf, "user_config_dir", lambda *args: recursive_dir)
            conf.store_account(
                authentication_token,
                filename="config.toml",
                location="user_config",
                **DEFAULT_KWARGS
            )

        filepath = os.path.join(recursive_dir, "config.toml")
        result = toml.load(filepath)
        assert result == EXPECTED_CONFIG


class TestSaveConfigToFile:
    """Tests for the store_account function."""

    def test_correct(self, tmpdir):
        """Test saving a configuration file."""
        filepath = str(tmpdir.join("config.toml"))

        conf.save_config_to_file(OTHER_EXPECTED_CONFIG, filepath)

        result = toml.load(filepath)
        assert result == OTHER_EXPECTED_CONFIG

    def test_file_already_existed(self, tmpdir):
        """Test saving a configuration file even if the file already
        existed."""
        filepath = str(tmpdir.join("config.toml"))

        with open(filepath, "w") as f:
            f.write(TEST_FILE)

        result_for_existing_file = toml.load(filepath)
        assert result_for_existing_file == EXPECTED_CONFIG

        conf.save_config_to_file(OTHER_EXPECTED_CONFIG, filepath)

        result_for_new_file = toml.load(filepath)
        assert result_for_new_file == OTHER_EXPECTED_CONFIG
