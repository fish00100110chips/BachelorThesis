"""
Check the status of a job using the Edge Impulse API.
https://docs.edgeimpulse.com/reference/edge-impulse-api/jobs/get_job_status
"""

import requests
import os
import argparse
from dotenv import load_dotenv # type: ignore
load_dotenv()

API_KEY = os.getenv("EI_API_KEY")
PROJECT_ID = os.getenv("EI_PROJECT_ID")

RUNNING = 0
SUCCESS = 1
FAILED = -1

def check_job_status(job_id, verbose=True):
    """
    Check the status of a job using the Edge Impulse API.
    https://docs.edgeimpulse.com/reference/edge-impulse-api/jobs/get_job_status

    Returns: 0 if still running
             1 if finished successfully
            -1 if failed
    """
    res = requests.get(
        f"https://studio.edgeimpulse.com/v1/api/{PROJECT_ID}/jobs/{job_id}/status",
        headers={ "x-api-key": API_KEY }
    )

    data = res.json()
    if res.status_code != 200:
        if verbose: print("❌ Failed to get job status:", res.status_code, res.text)
        return None

    job = data.get('job')
    if job:
        finished = job.get('finishedSuccessful')

        if finished is not None:
            if finished:
                if verbose:
                    print("✅ Job finished successfully.")
                    print("data", data)
                return SUCCESS
            else:
                if verbose: print("❌ Job failed.")
                return FAILED
        else:
            if verbose: print("⌛️ Job is still running or status unavailable.")
            return RUNNING
    else:
        if verbose: print("❗️ No job info found in response.")
        return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Check the status of a job.")
    parser.add_argument("job_id", type=int, help="The job ID to check.")
    args = parser.parse_args()
    print(check_job_status(args.job_id))