# Copyright 2010 Pallets

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

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
https://github.com/pallets/flask/blob/master/src/flask/logging.py
"""

import logging
import sys


def logging_handler_defined(logger):
    """Checks if the logger or any of its ancestors has a handler defined.

    The output depends on whether or not propagation was set for the logger.

    Args:
        logger (logging.Logger): the logger to check

    Returns:
        bool: whether or not a handler was defined
    """
    current = logger

    while current:
        if current.handlers:
            return True

        if not current.propagate:
            break

        current = current.parent

    return False


default_handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
default_handler.setFormatter(formatter)


def create_logger(name, level=logging.INFO):
    """Get the Strawberry Fields module specific logger and configure it if needed.

    Configuration only takes place if no user configuration was applied to the
    logger. Therefore, the logger is configured if and only if the following
    are true:

    - the logger has WARNING as effective level,
    - the level of the logger was not explicitly set,
    - no handlers were added to the logger.

    As the root logger has a WARNING level by default, any module specific
    logger will inherit the same as effective level.

    The default handler that is used for configuration writes to the standard
    error stream and uses a datetime and level formatter.

    Args:
        name (str): the name of the module for which the logger is being created
        level (logging.level): the logging level to set for the logger
    """
    logger = logging.getLogger(name)

    effective_level_inherited = logger.getEffectiveLevel() == logging.WARNING
    level_not_set = not logger.level
    no_handlers = not logging_handler_defined(logger)

    if effective_level_inherited and level_not_set and no_handlers:
        logger.setLevel(level)
        logger.addHandler(default_handler)

    return logger
