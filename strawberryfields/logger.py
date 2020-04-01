# Copyright 2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains functions for creating a logger that can be used in Strawberry
Fields.

The implementation in this module is based on the solution for logging used in
the Flask web application framework:
https://github.com/pallets/flask/blob/353d891561659a754ee92bb5e6576e82be58934a/src/flask/logging.py
"""

import logging
import sys

def has_level_handler(logger):
    """Check if there is a handler in the logging chain that will handle the
    given logger's :meth:`effective level <~logging.Logger.getEffectiveLevel>`.

    Args:
        logger (logging.Logger): the logger to check
    """
    level = logger.getEffectiveLevel()
    current = logger

    while current:
        if any(handler.level <= level for handler in current.handlers):
            return True

        if not current.propagate:
            break

        current = current.parent

    return False

default_handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
default_handler.setFormatter(formatter)


def create_logger(name, level=logging.DEBUG):
    """Get the Strawberry Fields module specific logger and configure it if needed.

    Args:
        name (str): the name of the module for which the logger is being created
        level (logging.level): the logging level to set for the logger
    """
    logger = logging.getLogger(name)

    if not logger.level:
        logger.setLevel(level)

    if not has_level_handler(logger):
        logger.addHandler(default_handler)

    return logger
