import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("EI_API_KEY")
PROJECT_ID = os.getenv("EI_PROJECT_ID")

headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

def delete_all_data():
    """
    Deletes all data in the Edge Impulse project.
    https://docs.edgeimpulse.com/reference/edge-impulse-api/rawdata/remove_all_samples
    https://studio.edgeimpulse.com/v1/api/{projectId}/raw-data/delete-all
    """
    response = requests.post(
        f"https://studio.edgeimpulse.com/v1/api/{PROJECT_ID}/raw-data/delete-all",
        headers=headers
    )

    if response.status_code == 200:
        print("All data deleted successfully.")
    else:
        print(f"Failed to delete data: {response.status_code} - {response.text}")


if __name__ == "__main__":
    delete_all_data()