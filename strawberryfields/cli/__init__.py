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

import argparse
import sys

from strawberryfields.api import Connection
from strawberryfields.configuration import ConfigurationError, create_config, store_account
from strawberryfields.engine import RemoteEngine
from strawberryfields.io import load


def main():
    """The Xanadu cloud platform command line interface.

    **Example:**

    The following is a simple example on getting the help message of the cloud platform command
    line interface. It details each of the options available.

    .. code-block:: console

        $ sf
        usage: sf [-h] [--ping] {configure,run} ...

        See below for available options and commands for working with the Xanadu cloud platform.

        General Options:
          -h, --help       show this help message and exit
          --ping, -p       Tests the connection to the remote backend.

        Commands:
          {configure,run}
            configure      Configure each detail of the API connection.
            run            Run a blackbird script.
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
        description="See below for available options and commands for working with the Xanadu cloud platform."
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
        "--local",
        "-l",
        action="store_true",
        help="Create a local configuration file in the current directory.",
    )

    # Adding the input subparser
    run_parser = subparsers.add_parser("run", help="Run a blackbird script.")
    run_parser.add_argument(
        "input", type=str, help="The filename or path to the blackbird script to run."
    )
    run_parser.set_defaults(func=run_blackbird_script)
    run_parser.add_argument(
        "--output",
        "-o",
        help="Path to the output file, where the results of the program will be written (stdout by default).",
    )

    return parser


def configure(args):
    r"""An auxiliary function for configuring the API connection to the Xanadu
    cloud platform.

    Can be used to simply save the authentication token with default
    configuration options. Alternatively, a wizard is provided for full
    configurability.

    See more details regarding Strawberry Fields configuration and available
    configuration options on the :doc:`/code/sf_configuration` page.

    Args:
        args (ArgumentParser): arguments that were specified on the command
            line stored as attributes in an argument parser object
    """
    if args.token:
        kwargs = {"authentication_token": args.token}
    else:
        kwargs = configuration_wizard()

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


def configuration_wizard():
    r"""Provides an interactive selection wizard on the command line to
    configure every option for the API connection.

    Default configuration options are provided as defaults to the user.
    Configuration options are detailed in :doc:`/code/sf_configuration`.

    Returns:
        dict[str, Union[str, bool, int]]: the configuration options
    """
    default_config = create_config()["api"]

    # Getting default values that can be used for as messages when getting inputs
    hostname_default = default_config["hostname"]
    ssl_default = "y" if default_config["use_ssl"] else "n"
    port_default = default_config["port"]

    authentication_token = input(
        "Please enter the authentication token to use when connecting: [] "
    )

    if not authentication_token:
        sys.stdout.write("No authentication token was provided, please configure again.")
        sys.exit()

    hostname = (
        input(
            "Please enter the hostname of the server to connect to: [{}] ".format(hostname_default)
        )
        or hostname_default
    )

    ssl_input = (
        input("Should the client attempt to connect over SSL? [{}] ".format(ssl_default))
        or ssl_default
    )
    use_ssl = ssl_input.upper() == "Y"

    port = (
        input("Please enter the port number to connect with: [{}] ".format(port_default))
        or port_default
    )

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

    eng = RemoteEngine(program.target)

    sys.stdout.write("Executing program on remote hardware...\n")
    result = eng.run(program)

    if result and result.samples is not None:
        write_script_results(result.samples, output_file=args.output)
    else:
        sys.stdout.write(
            "Ooops! Something went wrong with obtaining the results. Please check the Blackbird script specified and the connection to the remote engine."
        )
        sys.exit()


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
