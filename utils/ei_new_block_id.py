"""

This script retrieves a new block ID for an Edge Impulse project.
Needed for creating new impulses or adding blocks to existing ones.

https://docs.edgeimpulse.com/reference/edge-impulse-api/impulse/get_new_block_id
https://studio.edgeimpulse.com/v1/api/{projectId}/impulse/get-new-block-id
"""

import requests
import os
from dotenv import load_dotenv  # type: ignore
load_dotenv()

API_KEY = os.getenv("EI_API_KEY")
PROJECT_ID = os.getenv("EI_PROJECT_ID")

print("API_KEY:", API_KEY)
print("PROJECT_ID:", PROJECT_ID)


def new_block_id():
    """
    Get a new block ID from the Edge Impulse API.
    This is needed for creating new impulses or adding blocks to existing ones.
    """
    res = requests.post(
        f"https://studio.edgeimpulse.com/v1/api/{PROJECT_ID}/impulse/get-new-block-id",
        headers={"x-api-key": API_KEY}
    )
    print("Status Code:", res.status_code)
    print("Response Text:", res.text)

    data = res.json()
    if res.status_code == 200 and data.get('success', False):
        print("✅ New block ID retrieved:", data)
        return data.get('blockId', -1)
    else:
        print("❌ Failed to retrieve new block ID:", res.status_code, res.text)
        return -1

if __name__ == "__main__":
    new_block_id = new_block_id()
    print("New Block ID:", new_block_id)
