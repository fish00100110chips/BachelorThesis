"""
Classify job results into categories based on their content.
Can only be run if testing job has been run first.

https://studio.edgeimpulse.com/v1/api/{projectId}/classify/all/result
"""
import json
import os
import requests
from dotenv import load_dotenv  # type: ignore

load_dotenv()
API_KEY = os.getenv("EI_API_KEY")
PROJECT_ID = os.getenv("EI_PROJECT_ID")


def test_results(json_file="classification_result.json"):
    """
    To be run after the testing job has been run, to check the accuracy of the model.
    And to save the results to a file.
    Saves the full JSON response to the given json_file name
    and returns a simplified accuracy score of the classification job.
    """
    res = requests.get(
        f"https://studio.edgeimpulse.com/v1/api/{PROJECT_ID}/classify/all/result",
        headers={
            "x-api-key": API_KEY,
            "Content-Type": "application/json"
        }
    )

    if res.status_code == 200:

        # Save the full JSON response to a file
        data = res.json()
        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Classification result saved to {json_file}")

        # Return the accuracy score
        acc = data.get('accuracy', {})
        return acc.get('accuracyScore', -1)
    else:
        print("❌ Failed to classify job result:", res.status_code, res.text)
        return -1

if __name__ == "__main__":

    result = test_results("results_EXP1_FRONT_CHUNKED_160_transfer_efficientnet_b0_run0_chunk_6.json")
    print("Job classification result:", result)
