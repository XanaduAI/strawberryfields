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
This module contains functions used to load, store, save, and modify
configuration options for Strawberry Fields.
"""
import os

import toml
from appdirs import user_config_dir

from strawberryfields.logger import create_logger

DEFAULT_CONFIG_SPEC = {
    "api": {
        "authentication_token": (str, ""),
        "hostname": (str, "platform.strawberryfields.ai"),
        "use_ssl": (bool, True),
        "port": (int, 443),
    }
}


class ConfigurationError(Exception):
    """Exception used for configuration errors"""


def load_config(filename="config.toml", **kwargs):
    """Load configuration from keyword arguments, configuration file or
    environment variables.

    .. note::

        The configuration dictionary will be created based on the following
        (order defines the importance, going from most important to least
        important):

        1. keyword arguments passed to ``load_config``
        2. data contained in environmental variables (if any)
        3. data contained in a configuration file (if exists)

    Keyword Args:
        filename (str): the name of the configuration file to look for.
            Additional configuration options are detailed in
            :doc:`/code/sf_configuration`

    Returns:
        dict[str, dict[str, Union[str, bool, int]]]: the configuration
    """
    config = create_config()

    filepath = find_config_file(filename=filename)

    if filepath is not None:
        loaded_config = load_config_file(filepath)
        api_config = get_api_config(loaded_config, filepath)

        valid_api_options = keep_valid_options(api_config)
        config["api"].update(valid_api_options)
    else:
        log = create_logger(__name__)
        log.warning("No Strawberry Fields configuration file found.")

    update_from_environment_variables(config)

    valid_kwargs_config = keep_valid_options(kwargs)
    config["api"].update(valid_kwargs_config)

    return config


def create_config(authentication_token=None, **kwargs):
    """Create a configuration object that stores configuration related data
    organized into sections.

    The configuration object contains API-related configuration options. This
    function takes into consideration only pre-defined options.

    If called without passing any keyword arguments, then a default
    configuration object is created.

    Keyword Args:
        Configuration options as detailed in :doc:`/code/sf_configuration`

    Returns:
        dict[str, dict[str, Union[str, bool, int]]]: the configuration
            object
    """
    authentication_token = authentication_token or ""
    hostname = kwargs.get("hostname", DEFAULT_CONFIG_SPEC["api"]["hostname"][1])
    use_ssl = kwargs.get("use_ssl", DEFAULT_CONFIG_SPEC["api"]["use_ssl"][1])
    port = kwargs.get("port", DEFAULT_CONFIG_SPEC["api"]["port"][1])

    config = {
        "api": {
            "authentication_token": authentication_token,
            "hostname": hostname,
            "use_ssl": use_ssl,
            "port": port,
        }
    }
    return config


def delete_config(filename="config.toml", directory=None):
    """Delete a configuration file.

    If called with no arguments, the currently active configuration file is deleted.

    Keyword Args:
        filename (str): the filename of the configuration file to delete
        directory (str): the directory of the configuration file to delete
            If ``None``, the currently active configuration file is deleted.
    """
    if directory is None:
        file_path = find_config_file(filename)
    else:
        file_path = os.path.join(directory, filename)

    os.remove(file_path)


def reset_config(filename="config.toml"):
    """Delete all active configuration files

    .. warning::
        This will delete all configuration files with the specified filename
        (default ``config.toml``) found in the configuration directories.

    Keyword Args:
        filename (str): the filename of the configuration files to reset
    """
    for config in get_available_config_paths(filename):
        delete_config(os.path.basename(config), os.path.dirname(config))


def find_config_file(filename="config.toml"):
    """Get the filepath of the first configuration file found from the defined
    configuration directories (if any).

    .. note::

        The following directories are checked (in the following order):

        * The current working directory
        * The directory specified by the environment variable ``SF_CONF`` (if specified)
        * The user configuration directory (if specified)

    Keyword Args:
        filename (str): the configuration file to look for

    Returns:
         Union[str, None]: the filepath to the configuration file or None, if
             no file was found
    """
    directories = directories_to_check()
    for directory in directories:
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            return filepath

    return None


def directories_to_check():
    """Returns the list of directories that should be checked for a configuration file.

    .. note::

        The following directories are checked (in the following order):

        * The current working directory
        * The directory specified by the environment variable ``SF_CONF`` (if specified)
        * The user configuration directory (if specified)

    Returns:
        list: the list of directories to check
    """
    directories = []

    current_dir = os.getcwd()
    sf_env_config_dir = os.environ.get("SF_CONF", "")
    sf_user_config_dir = user_config_dir("strawberryfields", "Xanadu")

    directories.append(current_dir)
    if sf_env_config_dir != "":
        directories.append(sf_env_config_dir)
    directories.append(sf_user_config_dir)

    return directories


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


def get_api_config(loaded_config, filepath):
    """Gets the API section from the loaded configuration.

    Args:
        loaded_config (dict): the configuration that was loaded from the TOML config
            file
        filepath (str): path to the configuration file

    Returns:
        dict[str, Union[str, bool, int]]: the api section of the configuration

    Raises:
        ConfigurationError: if the api section was not defined in the
            configuration
    """
    try:
        return loaded_config["api"]
    except KeyError:
        log = create_logger(__name__)
        log.error('The configuration from the %s file does not contain an "api" section.', filepath)
        raise ConfigurationError


def keep_valid_options(sectionconfig):
    """Filters the valid options in a section of a configuration dictionary.

    Args:
        sectionconfig (dict[str, Union[str, bool, int]]): the section of the
            configuration to check

    Returns:
        dict[str, Union[str, bool, int]]: the keep section of the
            configuration
    """
    return {k: v for k, v in sectionconfig.items() if k in VALID_KEYS}


def update_from_environment_variables(config):
    """Updates the current configuration object from data stored in environment
    variables.

    The list of environment variables can be found at :mod:`strawberryfields.configuration`

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

    if DEFAULT_CONFIG_SPEC["api"][key][0] is bool:
        if value in trues:
            return True

        if value in falses:
            return False

        raise ValueError("Boolean could not be parsed")

    if DEFAULT_CONFIG_SPEC["api"][key][0] is int:
        return int(value)

    return value


def active_configs(filename="config.toml"):
    """Prints the filepaths for existing configuration files to the standard
    output and marks the one that is active.

    This function relies on the precedence ordering of directories to check
    when marking the active configuration.

    Args:
        filename (str): the name of the configuration files to look for
    """
    active_configs_list = get_available_config_paths(filename)

    # print the active configurations found based on the filename specified
    if active_configs_list:
        active = True

        print(
            "\nThe following Strawberry Fields configuration files were found "
            'with the name "{}":\n'.format(filename)
        )

        for config in active_configs_list:
            if active:
                config += " (active)"
                active = False

            print("* " + config)
    else:
        print(
            "\nNo Strawberry Fields configuration files were found with the "
            'name "{}".\n'.format(filename)
        )

    # print the directores that are being checked for a configuration file
    directories = directories_to_check()

    print("\nThe following directories were checked:\n")
    for directory in directories:
        print("* " + directory)


def get_available_config_paths(filename="config.toml"):
    """Get the paths for the configuration files available in Strawberry Fields.

    Args:
        filename (str): the name of the configuration files to look for

    Returns:
        list[str]: the filepaths for the active configurations
    """
    active_configs_list = []

    directories = directories_to_check()

    for directory in directories:
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            active_configs_list.append(filepath)

    return active_configs_list


def store_account(authentication_token, filename="config.toml", location="user_config", **kwargs):
    r"""Configure Strawberry Fields for access to the Xanadu cloud platform by
    saving your account credentials.

    The configuration file can be created in the following locations:

    - A global user configuration directory (``"user_config"``)
    - The current working directory (``"local"``)

    This global user configuration directory differs depending on the operating system:

    * On Linux: ``~/.config/strawberryfields``
    * On Windows: ``C:\Users\USERNAME\AppData\Local\Xanadu\strawberryfields``
    * On MacOS: ``~/Library/Application Support/strawberryfields``

    By default, Strawberry Fields will load the configuration and account credentials from the global
    user configuration directory, no matter the working directory. However, if there exists a configuration
    file in the *local* working directory, this takes precedence. The ``"local"`` option is therefore useful
    for maintaining per-project configuration settings.

    **Examples:**

    In these examples ``"AUTHENTICATION_TOKEN"`` should be replaced with a valid authentication
    token.

    Access to the Xanadu cloud can be configured as follows:

    >>> sf.store_account("AUTHENTICATION_TOKEN")

    This creates the following ``"config.toml"`` file:

    .. code-block:: toml

        [api]
        authentication_token = "AUTHENTICATION_TOKEN"
        hostname = "platform.strawberryfields.ai"
        use_ssl = true
        port = 443

    You can also create the configuration file locally (in the **current
    working directory**) the following way:

    >>> import strawberryfields as sf
    >>> sf.store_account("AUTHENTICATION_TOKEN", location="local")

    Each of the configuration options can be passed as further keyword
    arguments as well (see the :doc:`/code/sf_configuration` page
    for a list of options):

    >>> import strawberryfields as sf
    >>> sf.store_account("AUTHENTICATION_TOKEN", location="local", hostname="MyHost", use_ssl=False, port=123)

    This creates the following ``"config.toml"`` file in the **current working directory**:

    .. code-block:: toml

        [api]
        authentication_token = "AUTHENTICATION_TOKEN"
        hostname = "MyHost"
        use_ssl = false
        port = 123

    Args:
        authentication_token (str): API token for authentication to the Xanadu cloud platform.
            This is required for submitting remote jobs using :class:`~.RemoteEngine`.

    Keyword Args:
        location (str): determines where the configuration file should be saved
        filename (str): the name of the configuration file to look for

    Additional configuration options are detailed in :doc:`/code/sf_configuration` and can be passed
    as keyword arguments.
    """
    if location == "user_config":
        directory = user_config_dir("strawberryfields", "Xanadu")

        # Create target Directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
    elif location == "local":
        directory = os.getcwd()
    else:
        raise ConfigurationError("This location is not recognized.")

    filepath = os.path.join(directory, filename)

    config = create_config(authentication_token=authentication_token, **kwargs)
    save_config_to_file(config, filepath)


def save_config_to_file(config, filepath):
    """Saves a configuration to a TOML file.

    Args:
        config (dict[str, dict[str, Union[str, bool, int]]]): the
            configuration to be saved
        filepath (str): path to the configuration file
    """
    with open(filepath, "w") as f:
        toml.dump(config, f)


VALID_KEYS = set(create_config()["api"].keys())
DEFAULT_CONFIG = create_config()
