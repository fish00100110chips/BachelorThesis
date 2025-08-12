"""
Example script to classify an image using a model from Edge Impulse.
From: https://github.com/edgeimpulse/linux-sdk-python
"""
import cv2 # type: ignore
import os
import sys
import argparse
from edge_impulse_linux.image import ImageImpulseRunner # type: ignore


def run_model(modelfile, imgfile):

    runner = None

    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            # model_info = runner.init(debug=True) # to get debug print out

            print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
            labels = model_info['model_parameters']['labels']

            img = cv2.imread(imgfile)
            if img is None:
                print('Failed to load image', imgfile)
                exit(1)

            # imread returns images in BGR format, so we need to convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # get_features_from_image also takes a crop direction arguments in case you don't have square images
            # features, cropped = runner.get_features_from_image(img)

            # this mode uses the same settings used in studio to crop and resize the input
            features, cropped = runner.get_features_from_image_auto_studio_settings(img)

            res = runner.classify(features)

            if "classification" in res["result"].keys():
                print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
                for label in labels:
                    score = res['result']['classification'][label]
                    if score > 0.01:
                        print('%s: %.2f\t' % (label, score), end='')
                print('', flush=True)

            # Maybe relevant if we want to also detect objects in the image
            elif "bounding_boxes" in res["result"].keys():
                print('Found %d bounding boxes (%d ms.)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))
                for bb in res["result"]["bounding_boxes"]:
                    print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % (bb['label'], bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))
                    cropped = cv2.rectangle(cropped, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (255, 0, 0), 1)

            else:
                print('Are you sure this is a classification model?')
                exit(1)

            # the image will be resized and cropped, save a copy of the picture here
            # so you can see what's being passed into the classifier
            cv2.imwrite('debug.jpg', cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

        finally:
            if (runner):
                runner.stop()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Classify an image using a model from Edge Impulse.')
    parser.add_argument('model', type=str, help='Path to the model file (.eim)')
    parser.add_argument('image', type=str, help='Path to the image file (.jpg)')
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, args.model)
    imgfile = os.path.join(dir_path, args.image)

    if not os.path.isfile(modelfile) or not os.path.isfile(imgfile):
        print('Model or img file does not exist')
        sys.exit(2)

    run_model(modelfile, imgfile)
