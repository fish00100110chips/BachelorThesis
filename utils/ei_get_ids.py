"""
Helper functions to find the DSP and learn block IDs from the Edge Impulse API.
Needed for generating features and training the model, respectively.
Used in generate_features.py and ei_train.py / auto_train_download.py.
"""
import requests
import os
from dotenv import load_dotenv # type: ignore
load_dotenv()

API_KEY = os.getenv("EI_API_KEY")
PROJECT_ID = os.getenv("EI_PROJECT_ID")


def get_dsp_id():
    """
    Get the dsp ID from the Edge Impulse API.
    This is needed for generating the features, after which we can train the model.
    """
    res = requests.get(
        f"https://studio.edgeimpulse.com/v1/api/{PROJECT_ID}/impulse",
        headers={ "x-api-key": API_KEY }
    )

    data = res.json()
    for block in data["impulse"]["dspBlocks"]:
        print(f"ID: {block['id']} | Type: {block['type']} | Name: {block['name']}")
        return block["id"]
    return -1  # Not found


def learn_block_id():
    """
    Get the learn block ID from the Edge Impulse API.
    This is needed for training the model.
    """
    res = requests.get(
        f"https://studio.edgeimpulse.com/v1/api/{PROJECT_ID}/impulse",
        headers={ "x-api-key": API_KEY }
    )

    data = res.json()
    for block in data["impulse"]["learnBlocks"]:
        print(f"ID: {block['id']} | Type: {block['type']} | Name: {block['name']}")
        if block["type"] == "keras-transfer-image":
            return block["id"]
    return -1  # Not found

if __name__ == "__main__":
    block_id = learn_block_id()
    if block_id != -1:
        print(f"Learn block ID: {block_id}")
    else:
        print("Learn block ID not found.")

    dsp_id = get_dsp_id()
    if dsp_id != -1:
        print(f"DSP block ID: {dsp_id}")
    else:
        print("DSP block ID not found.")