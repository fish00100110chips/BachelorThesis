"""
Deletes the default impulse from our Edge Impulse project.
https://studio.edgeimpulse.com/v1/api/{projectId}/impulse
"""

import requests
import os
from dotenv import load_dotenv  # type: ignore
load_dotenv()
API_KEY = os.getenv("EI_API_KEY")
PROJECT_ID = os.getenv("EI_PROJECT_ID")

def delete_impulse():
    """
    Deletes the default impulse from our Edge Impulse project.
    """
    res = requests.delete(
        f"https://studio.edgeimpulse.com/v1/api/{PROJECT_ID}/impulse",
        headers={
            "x-api-key": API_KEY,
            "Content-Type": "application/json"
        }
    )

    if res.status_code == 200:
        print("✅ Impulse deleted successfully.")
    else:
        print("❌ Failed to delete impulse:", res.status_code, res.text)

if __name__ == "__main__":
    delete_impulse()