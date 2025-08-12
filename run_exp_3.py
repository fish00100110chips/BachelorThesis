"""
Runs the object orientation experiment.


"""

"""
Automates the data upload -> training -> testing -> downloading of the
Edge Impulse model.
"""
import os
import time
from dotenv import load_dotenv # type: ignore
load_dotenv()

from utils.ei_train import train_model
from utils.ei_train_efficientnet import train_efficientnet_model
from utils.job_status import check_job_status, RUNNING, SUCCESS, FAILED
from utils.ei_fetch_model import download_model_file
from utils.ei_get_ids import learn_block_id, get_dsp_id
from utils.ei_test_model import test_model
from utils.ei_test_results import test_results
from utils.ei_folder_upload import upload_dataset_folder
from utils.ei_generate_features import generate_features
from utils.ei_create_impulse import create_impulse
from utils.ei_delete_impulse import delete_impulse
from utils.ei_delete_all_data import delete_all_data
from dataset.prep_ds_exp1 import create_datasets_for_exp_1
from dataset.prep_ds_exp2 import create_datasets_for_exp_2

API_KEY = os.getenv("EI_API_KEY")
PROJECT_ID = os.getenv("EI_PROJECT_ID")


def wait_for_job_completion(job_id):
    """ Periodically (every min) checks the job status until it is finished."""

    start = time.time()
    status = check_job_status(job_id)

    while status == RUNNING:
        elapsed = time.time() - start
        print(f"Elapsed time: {elapsed // 60} minutes {round(elapsed % 60)} seconds")
        time.sleep(10)
        status = check_job_status(job_id)

    return status


def train_test_automation(model_type, save_name, img_size):
    """
    This function automates the process of training, testing, and downloading the Edge Impulse model.
    Useful for testing how accuracy scales with dataset size, and different model types.
    """

 #   Tamara's input: Check if the results file already exists, if so, skip the training and testing.
    json_file = f"results_{save_name}.json"
    if os.path.exists(json_file):
        print(f"Results for {save_name} already exist, skipping...")
        return

    print("Deleting old impulse...")
    delete_impulse()

    print("Creating impulse...")
    create_impulse(name="MyImpulse", img_size=img_size, dsp_type="image", model_name=model_type)
    print("Impulse created successfully.")

    print("Generating features for the dataset...")
    dsp_id = get_dsp_id()
    if dsp_id == -1:
        print("DSP block ID not found.")
        exit(1)
    print(f"DSP block ID: {dsp_id}")
    job_id = generate_features(dsp_id)
    wait_for_job_completion(job_id)

    block_id = learn_block_id()
    if block_id == -1:
        print("Learn block ID not found.")
        exit(1)
    print(f"Learn block ID: {block_id}")

    if model_type == "transfer_efficientnet_b0":
        job = train_efficientnet_model(block_id)
    else:
        job = train_model(block_id, model_type)
    job_id = job.get("id")
    if job_id is None:
        print("Job ID not found.")
        exit(1)
    print(f"Job ID: {job_id}")

    status = wait_for_job_completion(job_id)
    if status != SUCCESS:
        print("Job failed. Exiting.")
        exit(1)

    print("Testing the model...")
    test_job_id = test_model()
    status = wait_for_job_completion(test_job_id)
    if status != SUCCESS:
        print("Model testing failed. Exiting.")
        exit(1)

    # Use the given save_name and given accuracy to create a unique model and json file name
    json_file = f"results_{save_name}.json"

    acc_summary = test_results(json_file)
    print("Model accuracy summary:", acc_summary)
    print("Downloading model...")

    # modelname = f"model_{save_name}_acc_{round(acc_summary)}.eim"
    # download_model_file(modelname)


def run_exp_3():

    datasets_exp_2 = [
        "BASE/EXP1_FRONT_CHUNKED",
        "BASE/EXP1_SPLIT_CHUNKED",
        "BASE/EXP1_COMBINED_CHUNKED",
    ]

    model_types_96 = [
        # "transfer_mobilenetv2_a35", #0.35
        # "transfer_mobilenetv2_a1",  #0.1
        # "transfer_mobilenetv2_a05", #0.05

        # "transfer_mobilenetv1_a25_d100" #0.25
        # "transfer_mobilenetv1_a2_d100",  #0.2
        # "transfer_mobilenetv1_a1_d100"   #0.1
    ]

    model_types_160 = [
        # "transfer_mobilenetv2_160_a1",  #1.0   <-- deze gaat winnen wss
        "transfer_mobilenetv2_160_a75", #0.75
        # "transfer_efficientnet_b0",
        # "transfer_mobilenetv2_160_a5",  #0.5
        # "transfer_mobilenetv2_160_a35", #0.35
    ]

    iterations = 5

    for i in range(iterations):  # Multiple iterations, as it is nondeterministic

        print(i)
        if i > 0:
            create_datasets_for_exp_1(seed=i)

        for dataset in datasets_exp_2:
            print(f"Processing dataset: {dataset}")
            dataset_path = os.path.abspath(dataset)
            print(f"Running experiment on data: {dataset}")
            dataset_chunks = os.listdir(dataset_path)
            # Sort it, so that the datasets are processed in a consistent order
            dataset_chunks.sort()
            print(f"Found dataset_chunks: {dataset_chunks}")


            print("Deleting all data on Edge Impulse for new experiment...")
            delete_all_data()

            # Upload the entire dataset folder
            for chunk_num, ds_chunk in enumerate(dataset_chunks):
                print(f"Processing chunk {chunk_num} of dataset {dataset} for run {i}")
                ds_chunk_path = os.path.join(dataset_path, ds_chunk)
                upload_dataset_folder(ds_chunk_path)

            print(f"Dataset {dataset} uploaded successfully.")

            # img_size = 96
            # for model_type in model_types_96:
            #     save_name = f"{dataset.split('/')[-1]}_{img_size}_{model_type}_run{i}_chunk_{chunk_num}"
            #     print(f"Running experiment for {save_name} with image size {img_size}. RUN {i}")
            #     train_test_automation(model_type, save_name, img_size)

            img_size = 160
            for model_type in model_types_160:
                save_name = f"{dataset.split('/')[-1]}_{img_size}_{model_type}_run{i}_chunk_{chunk_num}"
                print(f"Running experiment for {save_name} with image size {img_size}. RUN {i}")
                train_test_automation(model_type, save_name, img_size)


if __name__ == "__main__":
    run_exp_3()
