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
r"""
This module contains the :class:`Configuration` class, which is used to
load, store, save, and modify configuration options for Strawberry Fields.
"""
import os
import logging as log

import toml
from appdirs import user_config_dir

log.getLogger()


DEFAULT_CONFIG = {
    "api": {
        "authentication_token": "",
        "hostname": "localhost",
        "use_ssl": True,
        "port": 443,
        "debug": False}
}

BOOLEAN_KEYS = ("debug", "use_ssl")


def parse_environment_variable(key, value):
    trues = (True, "true", "True", "TRUE", "1", 1)
    falses = (False, "false", "False", "FALSE", "0", 0)

    if key in BOOLEAN_KEYS:
        if value in trues:
            return True
        elif value in falses:
            return False
        else:
            raise ValueError("Boolean could not be parsed")
    else:
        return value


class ConfigurationError(Exception):
    """Exception used for configuration errors"""


# This function will be used by the Connection object
def load_config(name="config.toml", **kwargs):
    _name = name

    _config = create_config_object(**kwargs)

    _config = update_from_config_file(_config)
    _config = update_from_environmental_variables(_config)


def create_config_object(**kwargs):
    authentication_token = kwargs.get("authentication_token", "")
    hostname = kwargs.get("hostname", "localhost")
    use_ssl = kwargs.get("use_ssl", True)
    port = kwargs.get("port", 443)
    debug = kwargs.get("debug", False)

    config = {
        "api": {
            "authentication_token": authentication_token,
            "hostname": hostname,
            "use_ssl": use_ssl,
            "port": port,
            "debug": debug
            }
    }
    return config

def update_from_config_file(config):

    current_dir = os.getcwd()
    env_config_dir = os.environ.get("SF_CONF", "")
    user_config_dir = user_config_dir("strawberryfields", "Xanadu")

    # Search the current directory, the directory under environment
    # variable SF_CONF, and default user config directory, in that order.
    directories = [current_dir, env_config_dir, user_config_dir]
    for directory in directories:
        _filepath = os.path.join(directory, _name)
        try:
            config = load_config_file.load(_filepath)
        except FileNotFoundError:
            log.info("No Strawberry Fields configuration file found.")
            config = False

    if config:
        self.update_config()
    else:
        log.info("No Strawberry Fields configuration file found.")
        # TODO: add logic for parsing from environmental variables

    return config

def update_from_environmental_variables(_config):

# calling on the write_config_file function
def write_config_file(name="config.toml", path=_user_config_dir, **kwargs):

    # TODO: create a config object similar to DEFAULT_CONFIG
    save_config_file(path, config)

def update_config(_config):
    """Updates the configuration from either a loaded configuration
    file, or from an environment variable.

    The environment variable takes precedence."""
    for section, section_config in _config.items():
        env_prefix = "SF_{}_".format(section.upper())

        for key in section_config:
            # Environment variables take precedence
            env = env_prefix + key.upper()

            if env in os.environ:
                # Update from environment variable
                _config[section][key] = parse_environment_variable(env, os.environ[env])
            elif _config_file and key in _config_file[section]:
                # Update from configuration file
                _config[section][key] = _config_file[section][key]
    return _config

def load_config_file(filepath):
    """Load a configuration file.

    Args:
        filepath (str): path to the configuration file
    """
    with open(filepath, "r") as f:
        _config_file = toml.load(f)

    return _config_file

def save_config_file(filepath, config):
    """Save a configuration file.

    Args:
        filepath (str): path to the configuration file
        config (dict of str: dict of str: Union[boolean, str, float])
    """
    with open(filepath, "w") as f:
        toml.dump(config, f)

class Configuration:
    """Configuration class.

    This class is responsible for loading, saving, and storing StrawberryFields
    and plugin/device configurations.

    Args:
        name (str): filename of the configuration file.
        This should be a valid TOML file. You may also pass an absolute
        or a relative file path to the configuration file.
    """

    def __str__(self):
        return "{}".format(self._config)

    def __repr__(self):
        return "Strawberry Fields Configuration <{}>".format(self._filepath)

    def __init__(self, name="config.toml"):
        # Look for an existing configuration file
        self._config = DEFAULT_CONFIG
        self._config_file = {}
        self._filepath = None
        self._name = name
        self._user_config_dir = user_config_dir("strawberryfields", "Xanadu")
        self._env_config_dir = os.environ.get("SF_CONF", "")

        # Search the current directory, the directory under environment
        # variable SF_CONF, and default user config directory, in that order.
        directories = [os.getcwd(), self._env_config_dir, self._user_config_dir]
        for directory in directories:
            self._filepath = os.path.join(directory, self._name)
            try:
                config = self.load(self._filepath)
                break
            except FileNotFoundError:
                config = False

        if config:
            self.update_config()
        else:
            log.info("No Strawberry Fields configuration file found.")

    def update_config(self):
        """Updates the configuration from either a loaded configuration
        file, or from an environment variable.

        The environment variable takes precedence."""
        for section, section_config in self._config.items():
            env_prefix = "SF_{}_".format(section.upper())

            for key in section_config:
                # Environment variables take precedence
                env = env_prefix + key.upper()

                if env in os.environ:
                    # Update from environment variable
                    self._config[section][key] = parse_environment_variable(env, os.environ[env])
                elif self._config_file and key in self._config_file[section]:
                    # Update from configuration file
                    self._config[section][key] = self._config_file[section][key]

    def __getattr__(self, section):
        if section in self._config:
            return self._config[section]

        raise ConfigurationError("Unknown Strawberry Fields configuration section.")

    @property
    def path(self):
        """Return the path of the loaded configuration file.

        Returns:
            str: If no configuration is loaded, this returns ``None``."""
        return self._filepath

    def load(self, filepath):
        """Load a configuration file.

        Args:
            filepath (str): path to the configuration file
        """
        with open(filepath, "r") as f:
            self._config_file = toml.load(f)

        return self._config_file

    def save(self, filepath):
        """Save a configuration file.

        Args:
            filepath (str): path to the configuration file
        """
        with open(filepath, "w") as f:
            toml.dump(self._config, f)

configuration = load_config()
