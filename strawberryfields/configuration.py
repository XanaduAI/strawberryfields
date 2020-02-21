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
def load_config(filename="config.toml", **kwargs):

    config = create_config_object(**kwargs)

    config_file = look_for_config_file(filename=filename)

    if config_file is not None:
        config = update_config_from_config_file(config, config_file)
    else:
        log.info("No Strawberry Fields configuration file found.")

    config = update_config_from_environmental_variables(config)

    return config

def create_config_object(**kwargs):
    print('kwargs in create_config_object', kwargs)
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
            "debug": debug,
            "port": port
            }
    }
    return config

def update_config_from_config_file(config, config_file):

    # Here an example for sectionconfig is api
    for section, sectionconfig in config.items():
        for key in sectionconfig:
            if key in config_file[section]:
                # Update from configuration file
                config[section][key] = config_file[section][key]

    return config

def look_for_config_file(filename="config.toml"):
    # Search the current directory, the directory under environment
    # variable SF_CONF, and default user config directory, in that order.
    current_dir = os.getcwd()
    sf_env_config_dir = os.environ.get("SF_CONF", "")
    sf_user_config_dir = user_config_dir("strawberryfields", "Xanadu")

    directories = [current_dir, sf_env_config_dir, sf_user_config_dir]
    for directory in directories:
        filepath = os.path.join(directory, filename)
        try:
            config_file = load_config_file(filepath)
            break
        except FileNotFoundError:
            config_file = None

    # TODO: maybe we need a merge here?
    return config_file

def update_config_from_environmental_variables(config):
    for section, sectionconfig in config.items():
        env_prefix = "SF_{}_".format(section.upper())
        for key in sectionconfig:
            env = env_prefix + key.upper()
            if env in os.environ:
                config[section][key] = parse_environment_variable(env, os.environ[env])

def load_config_file(filepath):
    """Load a configuration file.

    Args:
        filepath (str): path to the configuration file
    """
    with open(filepath, "r") as f:
        config_file = toml.load(f)
    return config_file

#configuration = load_config()
