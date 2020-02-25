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
r"""
This module contains the :class:`Configuration` class, which is used to
load, store, save, and modify configuration options for Strawberry Fields.
"""
import logging as log
import os

import toml
from appdirs import user_config_dir

log.getLogger()


class ConfigurationError(Exception):
    """Exception used for configuration errors"""


def load_config(filename="config.toml", **kwargs):
    """Load configuration from keyword arguments, configuration file or
    environment variables.

    .. note::

        The configuration object (a nested dictionary) will be created based
        on the following (order defines the importance, going from most
        important to least important):

        1. keyword arguments passed to ``load_config``
        2. data contained in environmental variables (if any)
        3. data contained in a configuration file (if exists)

    Keyword arguments:
        filename (str): the name of the configuration file to look for
        authentication_token (str): the token to be used for user
            authentication
        hostname (str): the name of the host to connect to
        use_ssl (bool): specifies if requests should be sent using SSL
        port (int): the port to be used when connecting to the remote service
        debug (bool): determines if the debugging mode is requested

    Returns:
        dict[str, dict[str, Union[str, bool, int]]]: the configuration
            object
    """
    config = create_config_object()

    parsed_config, _ = look_for_config_in_file(filename=filename)

    if parsed_config is not None:
        update_with_other_config(config, other_config=parsed_config)
    else:
        log.info("No Strawberry Fields configuration file found.")

    update_from_environment_variables(config)

    config_from_keyword_arguments = {"api": kwargs}
    update_with_other_config(config, other_config=config_from_keyword_arguments)

    return config

def create_config_object(authentication_token="", **kwargs):
    """Create a configuration object that stores configuration related data
    organized into sections.

    The configuration object contains API-related configuration options. This
    function takes into consideration only pre-defined options.

    If called without passing any keyword arguments, then a default
    configuration object is created.

    Keyword arguments:
        authentication_token (str): the token to be used for user
            authentication
        hostname (str): the name of the host to connect to
        use_ssl (bool): specifies if requests should be sent using SSL
        port (int): the port to be used when connecting to the remote service
        debug (bool): determines if the debugging mode is requested

    Returns:
        dict[str, dict[str, Union[str, bool, int]]]: the configuration
            object
    """
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

def look_for_config_in_file(filename="config.toml"):
    """Looks for the first configuration file to be found at certain paths.

    .. note::

        The following directories are checked (in the following order):

        * The current working directory
        * The directory specified by the environment variable SF_CONF (if specified)
        * The user configuration directory (if specified)

    Keyword arguments:
        filename (str): the configuration file to look for

    Returns:
         dict[str, dict[str, Union[str, bool, int]]] or None: the
             configuration object that was loaded
    """

    # Search the current directory, the directory under environment
    # variable SF_CONF, and default user config directory, in that order.
    current_dir = os.getcwd()
    sf_env_config_dir = os.environ.get("SF_CONF", "")
    sf_user_config_dir = user_config_dir("strawberryfields", "Xanadu")

    directories = [current_dir, sf_env_config_dir, sf_user_config_dir]
    for directory in directories:
        filepath = os.path.join(directory, filename)
        try:
            parsed_config = load_config_file(filepath)
            break
        except FileNotFoundError:
            parsed_config, filepath = None, None

    return parsed_config, filepath

def load_config_file(filepath):
    """Load a configuration object from a TOML formatted file.

    Args:
        filepath (str): path to the configuration file

    Returns:
         dict[str, dict[str, Union[str, bool, int]]]: the configuration
            object that was loaded
    """
    with open(filepath, "r") as f:
        config_from_file = toml.load(f)
    return config_from_file

def update_with_other_config(config, other_config):
    """Updates the current configuration object with another one.

    Args:
        config (dict[str, dict[str, Union[str, bool, int]]]): the
            configuration to be updated
        other_config (dict[str, dict[str, Union[str, bool, int]]]): the
            configuration used for updating

    Returns:
        dict[str, dict[str, Union[str, bool, int]]]): the updated
            configuration
    """
    # Here an example for section is API
    for section, sectionconfig in config.items():
        for key in sectionconfig:
            if key in other_config[section]:
                config[section][key] = other_config[section][key]

def update_from_environment_variables(config):
    """Updates the current configuration object from data stored in environment
    variables.

    .. note::

        The following environment variables are checked:

        * SF_API_AUTHENTICATION_TOKEN
        * SF_API_HOSTNAME
        * SF_API_USE_SSL
        * SF_API_DEBUG
        * SF_API_PORT

    Args:
        config (dict[str, dict[str, Union[str, bool, int]]]): the
            configuration to be updated
    Returns:
        dict[str, dict[str, Union[str, bool, int]]]): the updated
        configuration
    """
    for section, sectionconfig in config.items():
        env_prefix = "SF_{}_".format(section.upper())
        for key in sectionconfig:
            env = env_prefix + key.upper()
            if env in os.environ:
                config[section][key] = parse_environment_variable(key, os.environ[env])


BOOLEAN_KEYS = ("debug", "use_ssl")
INTEGER_KEYS = ("port")

def parse_environment_variable(key, value):
    """Parse a value stored in an environment variable.

    Args:
        key (str): the name of the environment variable
        value (Union[str, bool, int]): the value obtained from the environment
            variable

    Returns:
        [str, bool, int]: the parsed value
    """
    trues = (True, "true", "True", "TRUE", "1", 1)
    falses = (False, "false", "False", "FALSE", "0", 0)

    if key in BOOLEAN_KEYS:
        if value in trues:
            return True

        if value in falses:
            return False

        raise ValueError("Boolean could not be parsed")

    if key in INTEGER_KEYS:
        return int(value)

    return value

DEFAULT_CONFIG = create_config_object()
configuration = load_config()
config_file_path = look_for_config_in_file()[1]
