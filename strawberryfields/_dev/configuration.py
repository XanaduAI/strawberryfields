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
Configuration
=============

**Module name:** :mod:`strawberryfields.configuration`

.. currentmodule:: strawberryfields.configuration

This module contains the :class:`Configuration` class, which is used to
load, store, save, and modify configuration options for Strawberry Fields.

Behaviour
--------

On first import, Strawberry Fields attempts to load the configuration file `config.toml`, by
scanning the following three directories in order of preference:

1. The current directory
2. The path stored in the environment variable ``SF_CONF``
3. The default user configuration directory:

   * On Linux: ``~/.config/strawberryfields``
   * On Windows: ``~C:\Users\USERNAME\AppData\Local\Xanadu\strawberryfields``
   * On MacOS: ``~/Library/Preferences/strawberryfields``

If no configuration file is found, a warning message will be displayed in the logs,
and all device parameters will need to be passed as keyword arguments when
loading the device.

The user can access the initialized configuration via `strawberryfields.config`, view the
loaded configuration filepath, print the configurations options, access and modify
them via keys, and save/load new configuration files.

Configuration files
-------------------

The configuration file `config.toml` uses the `TOML standard <https://github.com/toml-lang/toml>`_,
and has the following format:

.. code-block:: toml

    [api]
    # Options for the Strawberry Fields Cloud API
    authentatication_token = "071cdcce-9241-4965-93af-4a4dbc739135"
    hostname = "localhost"
    use_ssl = true

Summary of options
------------------

todo

Summary of methods
------------------

.. currentmodule:: strawberryfields.configuration.Configuration

.. autosummary::
    path
    load
    save

Code details
~~~~~~~~~~~~

.. currentmodule:: strawberryfields.configuration

"""
import os
import logging as log

import toml
from appdirs import user_config_dir

log.getLogger()


DEFAULT_CONFIG = {"api": {"authentatication_token": "", "hostname": "localhost", "use_ssl": True}}


class ConfigurationError(Exception):
    """Exception used for configuration errors"""


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
                    self._config[section][key] = os.environ[env]
                elif self._config_file:
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
