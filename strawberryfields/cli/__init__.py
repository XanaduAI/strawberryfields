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

"""A standalone command-line interface for configuring Strawberry Fields and connecting
to the Xanadu cloud platform.
"""

import sys
import argparse

from strawberryfields.api import Connection
from strawberryfields.engine import StarshipEngine
from strawberryfields.io import load
from strawberryfields.configuration import store_account, create_config, ConfigurationError


PROMPTS = {
    "authentication_token": "Please enter the authentication token to use when connecting: [{}] ",
    "hostname": "Please enter the hostname of the server to connect to: [{}] ",
    "port": "Please enter the port number to connect with: [{}] ",
    "use_ssl": "Should the client attempt to connect over SSL? [{}] ",
}


def main():
    """The Xanadu cloud platform command line interface.

    Commands:
        * run
        * configure
    """
    parser = create_parser()
    args = parser.parse_args()

    if args.ping:
        ping()
    elif hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

def create_parser():
    """Creates a parser to process the commands and arguments passed to the
    command line interface.

    Returns:
        ArgumentParser: an argument parser object that defines the related
        options
    """
    parser = argparse.ArgumentParser(
        usage="starship <command> [<args>]",
        description="These are common options when working on the Xanadu cloud platform.",
    )

    # Setting a title for the general options (requires setting a private
    # attribute)
    parser._optionals.title = "General Options"

    # Adding the pinging general option
    parser.add_argument(
        "--ping", "-p", action="store_true", help="Tests the connection to the remote backend."
    )

    # Adding subparsers configure and input
    subparsers = parser.add_subparsers(title="Commands")

    # Adding the configure subparser
    configure_parser = subparsers.add_parser(
        "configure", help="Configure each detail of the API connection."
    )
    configure_parser.set_defaults(func=configure)

    configure_parser.add_argument(
        "--token",
        "-t",
        type=str,
        help="Configure Strawberry Fields with your Xanadu cloud platform API token.",
    )
    configure_parser.add_argument(
        "--local", "-l", action="store_true", help="Create a local configuration file in the current directory."
    )

    # Adding the input subparser
    run_parser = subparsers.add_parser("run", help="Run a blackbird script.")
    run_parser.add_argument(
        "input", type=str, help="The input blackbird script to run.",
    )
    run_parser.set_defaults(func=run_blackbird_script)
    run_parser.add_argument(
        "--output",
        "-o",
        help="Path to the output file, where the results of the program will be written (stdout by default).",
    )

    return parser


def configure(args):
    """An auxiliary function for configuring the API connection to the Xanadu
    cloud platform.

    Supports configuration by:
    * only specifying the token and using default configuration options;
    * specifying every configuration option one by one.

    Related arguments:
    * token: the authentication token to use
    * local: whether or not to create the configuration file locally

    Args:
        args (ArgumentParser): arguments that were specified on the command
            line stored as attributes in an argument parser object
    """
    if args.token:
        kwargs = {"authentication_token": args.token}
    else:
        kwargs = configure_everything()

    if args.local:
        store_account(**kwargs, location="local")
    else:
        store_account(**kwargs)


def ping():
    """Tests the connection to the remote backend."""
    if Connection().ping():
        sys.stdout.write("You have successfully authenticated to the platform!\n")
    else:
        sys.stdout.write("There was a problem when authenticating to the platform!\n")
    sys.exit()


def configure_everything():
    """Provides an interactive selection wizard on the command line to
    configure every option for the API connection.

    Default configuration options are provided as defaults to the user.
    Configuration options as detailed in :doc:`/introduction/configuration`.

    Returns:
        dict[str, Union[str, bool, int]]: the configuration options
    """
    default_config = create_config()["api"]

    authentication_token = (
        input(PROMPTS["authentication_token"].format(default_config["authentication_token"]))
    )

    if not authentication_token:
        sys.stdout.write("No authentication token was provided, please configure again.")
        sys.exit()

    hostname = (
        input(PROMPTS["hostname"].format(default_config["hostname"])) or default_config["hostname"]
    )

    ssl_default = "y" if default_config["use_ssl"] else "n"
    ssl_input = input(PROMPTS["use_ssl"].format(ssl_default)) or ssl_default
    use_ssl = ssl_input.upper() == "Y"

    port = input(PROMPTS["port"].format(default_config["port"])) or default_config["port"]

    kwargs = {
        "authentication_token": authentication_token,
        "hostname": hostname,
        "use_ssl": use_ssl,
        "port": port,
    }
    return kwargs

def run_blackbird_script(args):
    """Run a blackbird script.

    Related arguments:
    * input: the input blackbird script to be run
    * output: the output file to store the results in (optional)

    Args:
        args (ArgumentParser): arguments that were specified on the command
            line stored as attributes in an argument parser object
    """
    try:
        program = load(args.input)
    except FileNotFoundError:
        sys.stdout.write("The {} blackbird script was not found.".format(args.input))
        sys.exit()

    eng = StarshipEngine(program.target)

    sys.stdout.write("Executing program on remote hardware...\n")
    result = eng.run(program)

    if result and result.samples is not None:
        write_script_results(result.samples, output_file=args.output)

def write_script_results(samples, output_file=None):
    """Write the results of the script either to a file or to the standard output.

    Args:
        samples (array[float]): array of samples
        output_file (str or None): the path to the output file, None if no output was defined
    """
    if output_file:
        with open(output_file, "w") as file:
            file.write(str(samples))
    else:
        sys.stdout.write(str(samples))
