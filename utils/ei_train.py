"""
Train a transfer learning model using the Edge Impulse API.
Works for Keras models. (E.g. imagenetv2, which is our standard model in this project)
https://docs.edgeimpulse.com/reference/edge-impulse-api/jobs/train_model_-keras
"""
import requests
import os
from utils.ei_get_ids import learn_block_id
from dotenv import load_dotenv # type: ignore
load_dotenv()

API_KEY = os.getenv("EI_API_KEY")
PROJECT_ID = os.getenv("EI_PROJECT_ID")



def train_model(learn_block_id, model_type):
    """
    Train a model using the Edge Impulse API.

    learn_block_id:     The ID of the learn block to train,
                        as retrieved from learn_block_id.py.
    model_type:         The type of keras transfer model to train.
                        Available options as per june 2025 in Edge Impulse are:
        transfer_mobilenetv2_a35
        transfer_mobilenetv2_a1
        transfer_mobilenetv2_a05
        transfer_mobilenetv2_160_a1
        transfer_mobilenetv2_160_a75
        transfer_mobilenetv2_160_a5
        transfer_mobilenetv2_160_a35
        transfer_mobilenetv1_a25_d100
        transfer_mobilenetv1_a2_d100
        transfer_mobilenetv1_a1_d100


    Works for Keras models. (E.g. imagenetv2, which is our standard model in this project)
    https://docs.edgeimpulse.com/reference/edge-impulse-api/jobs/train_model_-keras
    """
    if not learn_block_id:
        print("Learn block ID is required for training.")
        return

    res = requests.post(
        f"https://studio.edgeimpulse.com/v1/api/{PROJECT_ID}/jobs/train/keras/{learn_block_id}",
        headers={
            "x-api-key": API_KEY,
            "Content-Type": "application/json"
        },
        # Play around with these parameters, see in Edge Impulse what standard settings are.
        # Maybe include some of this data in report? eh idk if that's interesting enough
        json={
            "mode": "visual",
            "trainingCycles": 20,  # 20 is standard
            "learningRate": 0.0005,  # 0.0005 is standard
            "batchSize": 16,
            "trainTestSplit": 0.2,  # 20% is standard
            "autoClassWeights": False,  # Not needed, our classes are balanced.
            "visualLayers": [
                {
                "type": model_type,
                "neurons": 0,  # no final dense layer, just the transfer model. Cam be experimented with in future works.
                "dropoutRate": 0.1,  # Between 0 and 1. Default is 0.1.
                }
            ],
            "profileInt8": False  # Setting this to false saves a lot of time, not needed for us.
        }
    )

    if res.status_code == 200:
        print("✅ Training started:", res.json())
    else:
        print("❌ Failed to start training:", res.status_code, res.text)
    return res.json()


if __name__ == "__main__":
    learn_id = learn_block_id()
    if learn_id == -1:
        print("Learn block ID not found.")
        exit(1)
    train_model(learn_id, "efficientnet_v2b0")
