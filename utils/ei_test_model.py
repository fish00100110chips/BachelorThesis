import requests
import os
from dotenv import load_dotenv  # type: ignore
load_dotenv()
API_KEY = os.getenv("EI_API_KEY")
PROJECT_ID = os.getenv("EI_PROJECT_ID")


def test_model():
    """
    Starts the model testing job.
    """
    res = requests.post(
        f"https://studio.edgeimpulse.com/v1/api/{PROJECT_ID}/jobs/classify",
        headers={
            "x-api-key": API_KEY,
            "Content-Type": "application/json"
        },
        json={"dataset": "testing"}
    )

    if res.status_code == 200:
        data = res.json()
        print("✅ Model testing started:", res.json())
        return data.get('id', None)
    else:
        print("❌ Failed to start model testing:", res.status_code, res.text)
        return None


if __name__ == "__main__":
    test_model()
