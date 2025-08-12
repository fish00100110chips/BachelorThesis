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



def train_efficientnet_model(learn_block_id):
    """
    Train a model using the Edge Impulse API.

    learn_block_id:     The ID of the learn block to train,
                        as retrieved from learn_block_id.py.

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
        # the json as stalked from the network inspector.
        # not part of the official API documentation.
        json={
            "trainTestSplit": 0.2,
            "customValidationMetadataKey": "",
            "autoClassWeights": False,
            "profileInt8": False,
            "learningRate": 0.0005,
            "trainingCycles": 20,
            "visualLayers": [
                {
                    "type": "transfer_organization",
                    "organizationModelId": 6575
                }
            ],
            "augmentationPolicyImage": "none",
            "useLearnedOptimizer": False,
            "blockParameters": {},
            "customParameters": {
                "epochs": "30",
                "learning-rate": "0.001",
                "use-pretrained-weights": "true",
                "freeze-percentage-of-layers": "90",
                "last-layers": "dense: 32, dropout: 0.1",
                "data-augmentation": "",
                "model-size": "b0",
                "batch-size": "16",
                "early-stopping": "true",
                "early-stopping-patience": "5",
                "early-stopping-min-delta": "0.001"
            }
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
    train_efficientnet_model(learn_id)
