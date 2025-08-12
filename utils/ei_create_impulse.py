
# #!/usr/bin/env python3
# """
# Create an impulse *and* set Keras parameters (model, LR, cycles…),
# then start training — all via the Edge Impulse Python SDK.
# """
# import os
# import requests
# from dotenv import load_dotenv
# from ei_new_block_id import new_block_id           # your helper
# from edgeimpulse_api import LearnApi, JobsApi
# from edgeimpulse_api.models import SetKerasParameterRequest

# load_dotenv()
# API_KEY     = os.getenv("EI_API_KEY")
# PROJECT_ID  = os.getenv("EI_PROJECT_ID")

# HEADERS = {
#     "x-api-key": API_KEY,
#     "Content-Type": "application/json"
# }

# def _create_impulse(name="MyImpulse", img_w=96, img_h=96, dsp_type="image"):
#     """Return (success_flag, learn_block_id)"""
#     ib_id, dsp_id, lb_id = new_block_id(), new_block_id(), new_block_id()

#     payload = {
#         "name": name,
#         "inputBlocks": [{
#             "id": ib_id, "type": "image", "name": "in", "title": "InputBlock",
#             "imageWidth": img_w, "imageHeight": img_h,
#             "resizeMode": "squash", "resizeMethod": "squash",


#         }],
#         "dspBlocks": [{
#             "id": dsp_id, "type": dsp_type, "name": "dsp", "input": ib_id,
#             "axes": ["image"], "title": "MyDSPBlockTitle"
#         }],
#         "learnBlocks": [{
#             "id": lb_id, "type": "keras-transfer-image",
#             "name": "learn", "dsp": [dsp_id], "title": "LearnBlock"
#         }]
#     }

#     r = requests.post(
#         f"https://studio.edgeimpulse.com/v1/api/{PROJECT_ID}/impulse",
#         headers=HEADERS, json=payload)

#     ok = r.ok and r.json().get("success", False)
#     print(("✅" if ok else "❌"), "Create impulse:", r.json() if ok else r.text)
#     return ok, lb_id if ok else None


# def _set_keras_params(learn_block_id, model_name,
#                       cycles=10, lr=1e-3, batch=16):
#     """Save parameters on the learn‑block and return the request body."""
#     req = SetKerasParameterRequest.from_dict({
#         "mode": "visual",
#         "training_cycles": cycles,
#         "learning_rate": lr,
#         "batch_size": batch,
#         "train_test_split": 0.2,
#         "auto_class_weights": True,
#         "blockParameters": {
#             "type": "keras-transfer-image",          # <-- REQUIRED
#             "model": model_name,
#             # optional extras:
#             # "denseNeurons": 64,
#             # "dropout": 0.25
#         }
#     })

#     learn_api = LearnApi()
#     learn_api.set_keras(PROJECT_ID, learn_block_id, req)
#     return req


# def create_impulse_and_train(model_name="transfer_mobilenetv1_a1_d100"):
#     ok, lb_id = _create_impulse()
#     if not ok:
#         return

#     req = _set_keras_params(lb_id, model_name)

#     # ----- start the training job -----
#     jobs_api = JobsApi()
#     job = jobs_api.train_keras_job(PROJECT_ID, lb_id, req)
#     print("🚀  Training job started:", job)


# if __name__ == "__main__":
#     create_impulse_and_train("transfer_mobilenetv2_a35")   # change backbone here


"""
Creates an impulse. Done after adding data to the dataset.
https://docs.edgeimpulse.com/reference/edge-impulse-api/impulse/create_impulse
"""
import requests
import os
from utils.ei_new_block_id import new_block_id
from dotenv import load_dotenv  # type: ignore

load_dotenv()
API_KEY = os.getenv("EI_API_KEY")
PROJECT_ID = os.getenv("EI_PROJECT_ID")


def create_impulse(name="MyImpulse", img_size=96, dsp_type="image", model_name="transfer_mobilenetv1_a1_d100"):
    """
    Create an impulse in Edge Impulse with specified image dimensions and DSP type.
    This is typically done after adding data to the dataset.

        img_size: Width  and height of the input image.
        name: Name of the impulse. Arbitrary.
        dsp_type: Type of DSP block, either "image" or "raw".
    """
    iB_id = new_block_id()
    dsp_id = new_block_id()
    lB_id = new_block_id()

    res = requests.post(
        f"https://studio.edgeimpulse.com/v1/api/{PROJECT_ID}/impulse",
        headers={
            "x-api-key": API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "name": name,
            "inputBlocks": [
                {
                "id": iB_id,
                "type": "image",
                "name": "MyInputBlockName",  # I really don't care about these names.
                "title": "MyInputBlockTitle",
                "imageWidth": img_size,
                "imageHeight": img_size,
                "resizeMode": "squash",
                "resizeMethod": "squash",
                }
            ],
            "dspBlocks": [
                {
                "id": dsp_id,
                "type": dsp_type,  # OR "raw"
                "name": "MyDSPBlockName",
                "axes": ["image"],
                "input": iB_id,
                "title": "MyDSPBlockTitle",
                }
            ],
            "learnBlocks": [
                {
                "id": lB_id,
                "type": "keras-transfer-image",
                "model": model_name,  # Only used for easy identification in the json results
                "name": "MyLearnBlockName",
                "dsp": [dsp_id],
                "title": "MyLearnBlockTitle",
                }
            ]

        }
    )

    data = res.json()
    if res.status_code == 200 and data.get('success', False):
        print("✅ Impulse created successfully:", data)
    else:
        print("❌ Failed to create impulse:", res.status_code, res.text)


if __name__ == "__main__":
    create_impulse()



    #         "learnBlocks": [
    #   [
    #     {
    #       "augmentationPolicyImage": ["all", "none"],
    #       "learningRate": [0.0005],
    #       "trainingCycles": [20],
    #       "type": "keras-transfer-image",
    #       "model": [
    #         "transfer_mobilenetv2_a35",
    #         "transfer_mobilenetv2_a1",
    #         "transfer_mobilenetv2_a05",
    #         "transfer_mobilenetv1_a2_d100",
    #         "transfer_mobilenetv1_a1_d100",
    #         "transfer_mobilenetv1_a25_d100"
            #     "transfer_mobilenetv2_a35",
            # "transfer_mobilenetv2_a1",
            # "transfer_mobilenetv2_a05",
            # "transfer_mobilenetv1_a2_d100",
            # "transfer_mobilenetv1_a1_d100",
            # "transfer_mobilenetv1_a25_d100",
            # "transfer_mobilenetv2_160_a1",
            # "transfer_mobilenetv2_160_a75",
            # "transfer_mobilenetv2_160_a5",
            # "transfer_mobilenetv2_160_a35"
    #       ],
    #       "denseNeurons": [16, 64],
    #       "dropout": [0.1, 0.5]
    #     }