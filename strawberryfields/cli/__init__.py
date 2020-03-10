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
    parser = create_parser()
    args = parser.parse_args()

    if args.ping:
        ping()
    elif hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

def create_parser():
    # TODO
    parser = argparse.ArgumentParser(usage='starship <command> [<args>]', description="These are common options when working on the Xanadu cloud platform.")

    # Setting a title for the general options (requires setting a private
    # attribute)
    parser._optionals.title = 'General Options'

    subparsers = parser.add_subparsers(title='Commands')
    configure_parser = subparsers.add_parser('configure', help='configure each detail of the API connection')
    configure_parser.set_defaults(func=configure)

    configure_parser.add_argument(
        "--token", "-t", type=str, help="configure the token of the API connection and use defaults"
    )
    configure_parser.add_argument(
        "--local", "-l", action="store_true", help="create the configure for the project"
    )

    script_parser = subparsers.add_parser('input', help='run a blackbird script')
    script_parser.set_defaults(func=run_blackbird_script)
    script_parser.add_argument(
        "--output",
        "-o",
        help="specify the path to output the result of the program (stdout by default)",
    )
    # TODO: add --configure option
    # TODO: add --token option
    parser.add_argument(
        "--ping", "-p", action="store_true", help="tests the connection to the remote backend"
    )

    return parser

def configure(args):
    print(args)
    if args.token:
        kwargs = {'authentication_token': args.token}
    else:
        kwargs = configure_everything()

    if args.local:
        store_account(**kwargs, location="local")
    else:
        store_account(**kwargs)

    # elif args.reconfigure:
    #    reconfigure_everything()

    # run_blackbird_script(args.input, args.output)

def ping():
    """Tests the connection to the remote backend.
    """
# TODO
    connection.ping()
    sys.stdout.write("You have successfully authenticated to the platform!\n")
    sys.exit()

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

    kwargs = {'authentication_token': authentication_token, 'hostname': hostname, 'use_ssl': use_ssl, 'port': port}
    return kwargs

def run_blackbird_script(args_input, args_output=None):
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

