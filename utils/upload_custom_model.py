"""
My attempt at uploading a custom model to edge impulse

POST /v1/api/{projectId}/training/keras/{learnId}
Host: studio.edgeimpulse.com
x-api-key: YOUR_API_KEY
Content-Type: application/json

{
  "mode": "expert",
  "script": "import tensorflow as tf\n\ndef build_model(model_input_shape, num_classes):\n    base_model = tf.keras.applications.MobileNetV2(\n        input_shape=model_input_shape,\n        include_top=False,\n        weights='imagenet'\n    )\n    base_model.trainable = False  # Freeze base layers\n\n    model = tf.keras.Sequential([\n        tf.keras.layers.InputLayer(input_shape=model_input_shape),\n        base_model,\n        tf.keras.layers.GlobalAveragePooling2D(),\n        tf.keras.layers.Dense(128, activation='relu'),\n        tf.keras.layers.Dropout(0.3),\n        tf.keras.layers.Dense(num_classes, activation='softmax')\n    ])\n\n    model.compile(\n        optimizer='adam',\n        loss='categorical_crossentropy',\n        metrics=['accuracy']\n    )\n    return model",
  "trainingCycles": 25,
  "learningRate": 0.001,
  "batchSize": 32,
  "trainTestSplit": 0.8,
  "autoClassWeights": true
}

"""
from ei_get_ids import learn_block_id, get_dsp_id
import requests
import os
from dotenv import load_dotenv # type: ignore
load_dotenv()

API_KEY = os.getenv("EI_API_KEY")
PROJECT_ID = os.getenv("EI_PROJECT_ID")

def upload_custom_model(learn_id, script, training_cycles=25, learning_rate=0.001, batch_size=32, train_test_split=0.8, auto_class_weights=True):
    """
    Upload a custom model script to Edge Impulse for training.

    learn_id: The ID of the learn block to train.
    script: The Python script defining the model architecture and training process.
    training_cycles: Number of training cycles.
    learning_rate: Learning rate for the optimizer.
    batch_size: Batch size for training.
    train_test_split: Proportion of data to use for training vs. testing.
    auto_class_weights: Whether to automatically compute class weights.
    """

    url = f"https://studio.edgeimpulse.com/v1/api/{PROJECT_ID}/training/keras/{learn_id}"

    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "mode": "expert",
        "script": script,
        "trainingCycles": training_cycles,
        "learningRate": learning_rate,
        "batchSize": batch_size,
        "trainTestSplit": train_test_split,
        "autoClassWeights": auto_class_weights
    }

    response = requests.post(url, headers=headers, json=payload)
    print(response.text)

    if response.status_code == 200:
        print("Model uploaded successfully.")
        return response.json()
    else:
        print(f"Failed to upload model: {response.status_code} - {response.text}")
        return None
#         return None
    return None


def get_script_from_file(file_path):
    """
    Read the Python script from a file.
    """
    with open(file_path, 'r') as file:
        script = file.read()
    return script

if __name__ == "__main__":
    # Example usage
    learn_id = learn_block_id()
    script = get_script_from_file("../../model_scripts/example_script.py")
    print(script)
    response = upload_custom_model(learn_id, script)
