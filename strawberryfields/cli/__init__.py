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

"""A standalone command-line interface for computing quantum programs on a remote
backend.
"""

import sys
import argparse

from strawberryfields.api import Connection
from strawberryfields.api.connection import connection
from strawberryfields.engine import StarshipEngine
from strawberryfields.io import load
from strawberryfields.configuration import store_account, configuration

def main():
    # TODO
    """: """
    args = parse_arguments()

    if args.ping:
        ping()
    elif args.token:
        configure_token(args.token)
    elif args.configure:
        configure_everything()

    run_program(args.input, args.output)

def parse_arguments():
    # TODO
    parser = argparse.ArgumentParser(description="These are common options when working on the Xanadu cloud platform.")
    group = parser.add_mutually_exclusive_group(required=True)
    # TODO: add --configure option
    # TODO: add --token option
    group.add_argument("--input", "-i", help="Path for the blackbird (.xbb) file to run.")
    group.add_argument(
        "--ping", "-p", action="store_true", help="test the API connection"
    )
    group.add_argument(
        "--token", "-t", action="store_true", help="configure the token of the API connection"
    )
    group.add_argument(
        "--configure", "-c", action="store_true", help="configure every detail of the API connection"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="where to output the result of the program - outputs to stdout by default",
    )

    return parser.parse_args()

def configure_token(authentication_token):
    store_account(authentication_token=authentication_token)

def configure_everything():

    default_config = configuration

    authentication_token = (
        input(PROMPTS["authentication_token"].format(default_config["authentication_token"]))
        or default_config["authentication_token"]
    )
    hostname = (
        input(PROMPTS["hostname"].format(default_config["hostname"])) or default_config["hostname"]
    )

    use_ssl = input(PROMPTS["use_ssl"].format("y" if default_config["use_ssl"] else "n")).upper() == "Y"

    port = input(PROMPTS["port"].format(default_config["port"])) or default_config["port"]

    store_account(authentication_token=authentication_token, hostname=hostname, use_ssl=use_ssl, port=port)

def ping():
    # TODO
        connection.ping()
        sys.stdout.write("You have successfully authenticated to the platform!\n")
        sys.exit()

def run_program(args_input, args_output=None):
    # TODO
    program = load(args_input)

    eng = StarshipEngine(program.target)
    sys.stdout.write("Executing program on remote hardware...\n")
    result = eng.run(program)

    if result and result.samples is not None:
        if args.output:
            with open(args_output, "w") as file:
                file.write(str(result.samples))
        else:
            sys.stdout.write(str(result.samples))

