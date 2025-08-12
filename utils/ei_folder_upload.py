"""
Uses the Edge Impulse CLI to upload multiple images to the Edge Impulse platform.

(De standaard ingestion API lukte me even niet, dus vandaar nu met de CLI)
https://docs.edgeimpulse.com/docs/tools/edge-impulse-cli/cli-uploader

# Jpg en PNG hoort gewoon te kunnen, zie:
https://docs.edgeimpulse.com/reference/data-ingestion/ingestion-api
Als je er zin in hebt kan je dit mooier maken ooit, maar het komt op hetzelfde neer,
the edge-impulse-uploader stuurt het toch ook door naar de Ingestion API.
Tip: use dataset/train_test_split.py
"""

import subprocess
import os
import argparse
from dotenv import load_dotenv # type: ignore
load_dotenv()

API_KEY = os.getenv("EI_API_KEY")


def upload_dataset_folder(dir_path):
    """
    Uploads a folder of images to Edge Impulse using the CLI uploader.
    Expected folder structure (as can be created with dataset/train_test_split.py):
    dir_path/
    - testing/
        - class1.0.jpg
        - class1.1.jpg
        - ...
        - class2.0.jpg
        - class2.1.jpg
        - ...
    - training/
        - class1.5.jpg
        - class1.6.jpg
        - ...
        - class2.9.jpg
        - class2.8.jpg
        - ...
    """

    if not os.path.isdir(dir_path):
        print(f"Error: The directory {dir_path} does not exist or is not a directory.")
        exit(1)

    command = [
        "edge-impulse-uploader",
        "--api-key", API_KEY,
        # "--category", "split",  # this does not work? Do manually
        # "--label", LABEL,  # leaving this out will infer label from the image name
        "--directory", dir_path
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Upload images from a folder to Edge Impulse.")
    parser.add_argument("directory", type=str, help="Path to the directory containing images to upload.")
    args = parser.parse_args()

    upload_dataset_folder(args.directory)
