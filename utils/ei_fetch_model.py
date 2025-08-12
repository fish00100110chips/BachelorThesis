"""
A python script to make it easier to download the model file from Edge Impulse.
Uses the CLI-based linux runner to download the model file.

Usage:
python3 ei_fetch_model.py --modelname <modelname.eim> --clean

Can also manually be done with:

edge-impulse-linux-runner --download modelfile.eim --api-key <KEY>

--clean : This flag allows you to log in with you edge impulse account and
          choose a different project again
--api-key : This is the API key for the project you want to download the model from.
            If you don't specify this, you'll be asked to log in with your Edge Impulse account.

We do it CLI-based, as .eim files are not available through the API as far as I know,
so it's not possible like in this documentation, I think:
https://docs.edgeimpulse.com/docs/tutorials/api-examples/running-jobs-through-the-python-sdk
Can be explored further if needed, but for now this is the easiest way to do it.

CAUTION: This is very dependent on the CLI version, if the CLI changes, this may need changing.
"""
import sys
import subprocess
import argparse
import os
from dotenv import load_dotenv # type: ignore
load_dotenv()

API_KEY = os.getenv("EI_API_KEY")


def check_ei_cli_version():
    """Returns the version of the Edge Impulse CLI."""
    try:
        result = subprocess.run(
            ["edge-impulse-linux-runner", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        cli_version = result.stdout.strip()
        if cli_version != '1.15.1':
            print(f"❗ ei_fetch_model.py was intented to work with \
                   Edge Impulse CLI version 1.15.1 not {cli_version}. \
                   Please check if this version works as well, or install the \
                   correct version. Proceed with caution.")
    except subprocess.CalledProcessError as e:
        print("❌ Failed to get CLI version:", e.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Edge Impulse CLI is not installed or not in PATH.")
        sys.exit(1)


def run_command(command):
    """Run a command in the command line and return the output."""
    try:
        result = subprocess.run(command, shell=True)

    except subprocess.CalledProcessError as e:
        print(f"Command failed with error:\n{e.stderr.decode('utf-8')}")
        sys.exit(1)


def download_model_file(modelname="model.eim", clean=False):
    """
    Download the model file from Edge Impulse.
    CAUTION: This is very dependent on the CLI version, if the CLI changes, this may need changing
    Can only work on projects with only one impulse!!!
    """
    # Make sure the Edge Impulse CLI is installed warn user if not the correct version.
    check_ei_cli_version()

    if not modelname.endswith(".eim"):
        print("Model name must end with .eim")
        sys.exit(1)

    command = "edge-impulse-linux-runner --download " + modelname
    command += " --api-key " + API_KEY  # For log in authentication.
    if clean:
        command += " --clean --api-key " + API_KEY

    # The yes command is a utility that repeatedly outputs a string to stout
    # until it's killed or interrupted.
    # This is done to automatically choose the unoptimized model file
    # CAUTION: IS VERY DEPENDENT ON THE CLI VERSION, IF THE CLI CHANGES, THIS MAY BREAK
    command = "yes '' | " + command

    run_command(command)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download model file from Edge Impulse.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="To log in with your Edge Impulse account and select a different project.",
    )
    parser.add_argument(
        "--modelname",
        type=str,
        default="model.eim",
        help="The name the model file will be downloaded as. Default is 'model.eim'.",
    )
    args = parser.parse_args()

    download_model_file(modelname=args.modelname, clean=args.clean)
