"""
Generate features for the dataset.
Needed before training the model.

https://docs.edgeimpulse.com/reference/edge-impulse-api/jobs/generate_features
https://studio.edgeimpulse.com/v1/api/{projectId}/jobs/generate-features

As can be done in the Edge Impulse dashboard:
https://studio.edgeimpulse.com/studio/712900/impulse/1/dsp/image/5
"""

import requests
import os
from utils.ei_get_ids import get_dsp_id
from dotenv import load_dotenv  # type: ignore
load_dotenv()

API_KEY = os.getenv("EI_API_KEY")
PROJECT_ID = os.getenv("EI_PROJECT_ID")


def generate_features(dsp_id):
    """
    Generate features for the dataset using the Edge Impulse API.
    This is needed before training the model.
    """
    res = requests.post(
        f"https://studio.edgeimpulse.com/v1/api/{PROJECT_ID}/jobs/generate-features",
        headers={
            "x-api-key": API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "dspId": dsp_id,
            "calculateFeatureImportance": True,
            "skipFeatureExplorer": False
        }
    )

    data = res.json()
    job_id = data.get('id', None)
    if res.status_code == 200 and data.get('success', False):
        print("✅ Feature generation started:", res.json())
    else:
        print("❌ Failed to start feature generation:", res.status_code, res.text)
    return job_id


if __name__ == "__main__":
    dsp_id = get_dsp_id()
    print(generate_features(dsp_id))
