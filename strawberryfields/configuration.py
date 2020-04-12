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
import collections
import os

import toml
from appdirs import user_config_dir


DEFAULT_CONFIG_SPEC = {
    "api": {
        "authentication_token": (str, ""),
        "hostname": (str, "platform.strawberryfields.ai"),
        "use_ssl": (bool, True),
        "port": (int, 443),
    },
    "logging": {"level": (str, "info"), "logfile": ((str, type(None)), None)},
}
"""dict: Nested dictionary representing the allowed configuration
sections, options, default values, and allowed types for Strawberry
Fields configurations. For each configuration option key, the
corresponding value is a length-2 tuple, containing:

* A type or tuple of types, representing the allowed type
  for that configuration option.

* The default value for that configuration option.

.. note::

    By TOML convention, keys with a default value of ``None``
    will **not** be present in the generated/loaded configuration
    file. This is because TOML has no concept of ``NoneType`` or ``Null``,
    instead, the non-presence of a key indicates that the configuration
    value is not set.
"""


class ConfigurationError(Exception):
    """Exception used for configuration errors"""


def _deep_update(source, overrides):
    """Recursively update a nested dictionary.

    This function is a generalization of Python's built in
    ``dict.update`` method, modified to recursively update
    keys with nested dictionaries.
    """
    for key, value in overrides.items():
        if isinstance(value, collections.Mapping) and value:
            # Override value is a non-empty dictionary.
            # Update the source key with the override dictionary.
            returned = _deep_update(source.get(key, {}), value)
            source[key] = returned
        elif value != {}:
            # Override value is not an empty dictionary.
            source[key] = overrides[key]
    return source


def _generate_config(config_spec, **kwargs):
    """Generates a configuration, given a Strawberry Fields configuration
    specification.

    See :attr:`~.DEFAULT_CONFIG_SPEC` for an example of a valid configuration
    specification.

    Optional keyword arguments may be provided to override default values
    in the cofiguration specification. If the provided override values
    do not match the expected type defined in the configuration spec,
    a ``ConfigurationError`` is raised.

    **Example**

    >>> _generate_config(DEFAULT_CONFIG_SPEC, api={"port": 54})
    {
        "api": {
            "authentication_token": "",
            "hostname": "platform.strawberryfields.ai",
            "use_ssl": True,
            "port": 54,
        }
    }

    Args:
        config_spec (dict): nested dictionary representing the
            configuration specification

    Keyword Args:
        Provided keyword arguments may overwrite default values of
        matching keys.

    Returns:
        dict: the default configuration defined by the input config spec

    Raises:
        ConfigurationError: if provided keyword argument overrides do not
        match the expected type defined in the configuration spec.
    """
    res = {}
    for k, v in config_spec.items():
        if isinstance(v, tuple):
            # config spec value v represents the allowed type and default value

            if k in kwargs:
                # Key also exists as a keyword argument.
                # Perform type validation.
                if not isinstance(kwargs[k], v[0]):
                    raise ConfigurationError(
                        "Expected type {} for option {}, received {}".format(
                            v[0], k, type(kwargs[k])
                        )
                    )

                if kwargs[k] is not None:
                    # Only add the key to the configuration object
                    # if the provided override is not None.
                    res[k] = kwargs[k]
            else:
                if v[1] is not None:
                    # Only add the key to the configuration object
                    # if the default value is not None.
                    res[k] = v[1]

        elif isinstance(v, dict):
            # config spec value is a dictionary of more options
            res[k] = _generate_config(v, **kwargs.get(k, {}))
    return res


def load_config(filename="config.toml", logging=True, **kwargs):
    """Load configuration from keyword arguments, configuration file or
    environment variables.

    .. note::

        The configuration dictionary will be created based on the following
        (order defines the importance, going from most important to least
        important):

        1. keyword arguments passed to ``load_config``
        2. data contained in environmental variables (if any)
        3. data contained in a configuration file (if exists)

    Args:
        filename (str): the name of the configuration file to look for
        logging (bool): whether or not to log details

    Keyword Args:
        Additional configuration options are detailed in
            :doc:`/code/sf_configuration`

    Returns:
        dict[str, dict[str, Union[str, bool, int]]]: the configuration
    """
    filepath = find_config_file(filename=filename)

    if logging:
        from strawberryfields.logger import create_logger

        log = create_logger(__name__)

    if filepath is not None:
        # load the configuration file
        with open(filepath, "r") as f:
            config = toml.load(f)

        if logging:
            log.debug("Configuration file %s loaded", filepath)

        if "api" not in config and logging:
            # Raise a warning if the configuration doesn't contain
            # an API section.
            log.warning(
                'The configuration from the %s file does not contain an "api" section.', filepath
            )

    else:
        config = {}

        if logging:
            log.warning("No Strawberry Fields configuration file found.")

    # update the configuration from environment variables
    update_from_environment_variables(config)

    # update the configuration from keyword arguments
    for config_section, section_options in kwargs.items():
        _deep_update(config, {config_section: section_options})

    # generate the configuration object by using the defined
    # configuration specification at the top of the file
    config = _generate_config(DEFAULT_CONFIG_SPEC, **config)

    # Log the loaded configuration details, masking out the API key.
    if logging:
        config_details = "Loaded configuration: {}".format(config)
        auth_token = config.get("api", {}).get("authentication_token", "")
        config_details = config_details.replace(auth_token[5:], "*" * len(auth_token[5:]))
        log.debug(config_details)

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
    directories = get_available_config_paths(filename=filename)

    if directories:
        return directories[0]

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

    if sf_env_config_dir:
        directories.append(sf_env_config_dir)

    directories.append(sf_user_config_dir)

    return directories


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
                config[section][key] = _parse_environment_variable(section, key, os.environ[env])


def _parse_environment_variable(section, key, value):
    """Parse a value stored in an environment variable.

    Args:
        section (str): configuration section name
        key (str): the name of the environment variable
        value (Union[str, bool, int]): the value obtained from the environment
            variable

    Returns:
        [str, bool, int]: the parsed value
    """
    trues = (True, "true", "True", "TRUE", "1", 1)
    falses = (False, "false", "False", "FALSE", "0", 0)

    if DEFAULT_CONFIG_SPEC[section][key][0] is bool:
        if value in trues:
            return True

        if value in falses:
            return False

        raise ValueError("Boolean could not be parsed")

    if DEFAULT_CONFIG_SPEC[section][key][0] is int:
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

    config = {}

    # load the existing config if it already exists
    if os.path.isfile(filepath):
        with open(filepath, "r") as f:
            config = toml.load(f)

    # update the loaded configuration file with the specified
    # authentication token
    kwargs.update({"authentication_token": authentication_token})

    # update the loaded configuration with any
    # provided API options passed as keyword arguments
    _deep_update(config, {"api": kwargs})

    # generate the configuration object by using the defined
    # configuration specification at the top of the file
    config = _generate_config(DEFAULT_CONFIG_SPEC, **config)

    with open(filepath, "w") as f:
        toml.dump(config, f)


DEFAULT_CONFIG = _generate_config(DEFAULT_CONFIG_SPEC)
SESSION_CONFIG = load_config(logging=False)
