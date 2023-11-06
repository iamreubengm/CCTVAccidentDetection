from __future__ import division
import math
import time
import numpy as np
import cv2
import os, sys
import random
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
import firebase_admin
from firebase_admin import credentials, firestore, storage
import signal
from darknet_images import *

enable_gpu = False

# def parser():
#     parser = argparse.ArgumentParser(description="YOLO Object Detection")
#     parser.add_argument("--path", type=str, default="../accident_vids/V1.mp4",
#                         help="Path to video file")
#     return parser.parse_args()


if enable_gpu:
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
else:
    # Disable all GPUS for tensorflow, comment if VRAM > 4GB
    try:
        tf.config.set_visible_devices([], "GPU")
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != "GPU"
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass




def send_to_db(image_path, Longitude=77, Latitude=13):
    name = int(time.time()*1000)
    bucket = storage.bucket()
    blob = bucket.blob(str(name))
    blob.upload_from_filename(image_path)
    blob.make_public()

    url = blob.public_url

    db = firestore.client()

    ref = db.collection("accidents")
    print(url)

    ref.document(str(time.time())).set(
        {
            "URL": url,
            "Longitude": Longitude,
            "Latitude": Latitude,
            "is_dismissed": False,
            "is_reported": False,
            "timestamp": name,
        }
    )


def extract_features(coloured_image_array):
    global last_accident
    current_accident = time.time()
    if (current_accident - last_accident) >= 5:

        # to prevent multiple frames of the same accident from being sent to db
        image_array = cv2.cvtColor(coloured_image_array, cv2.COLOR_BGR2GRAY)
        image_array = cv2.resize(image_array, (224, 224))
        image_array = cv2.merge([image_array,image_array,image_array])
        x = image.img_to_array(image_array)

        x = np.expand_dims(x, axis=0)
        x = np.array(x, dtype="float64")
        x = preprocess_input(x)


        features = model_new.predict(x, verbose = 0)
        features = features.flatten()

        S = [features]
        S = np.asarray(S)
        x = model_prediction.predict(S, verbose = 0)
        print(x)

        if x >= 0.84:
            print("Accident Occured")
            cv2.imwrite(
                f"accident_images/syays_{current_accident}.jpg", coloured_image_array
            )
            last_accident = current_accident
            pid = os.fork()
            if pid == 0: # Enters only for the child process
                send_to_db(f"accident_images/syays_{current_accident}.jpg")
                sys.exit(0)
                print('Process didnexitted')
            


# Checking for collision between objects from bounding box
def checkCollisions(value):
    for i in range(len(x_y_values)):
        for j in range(i + 1, len(x_y_values)):
            box1 = x_y_values[i]
            box2 = x_y_values[j]
            if proximityCheck(box1, box2):
                crop(box1, box2, value)


# Checking for proximity between ojects from bounding box
def proximityCheck(b1, b2):

    b1x = b1[4][0] + (b1[4][1] / 2)
    b2x = b2[4][0] + (b2[4][1] / 2)

    b1y = b1[5][0] + (b1[5][1] / 2)
    b2y = b2[5][0] + (b2[5][1] / 2)

    halfWidthSumX = (b1[4][1] / 2) + (b2[4][1]) / 2
    halfWidthSumY = (b1[5][1] / 2) + (b2[5][1]) / 2

    distBetweenCentersX = abs(b1x - b2x)
    distBetweenCentersY = abs(b1y - b2y)

    differenceX = distBetweenCentersX - halfWidthSumX
    differenceY = distBetweenCentersY - halfWidthSumY

    # if boxes completely overlap
    if differenceX < 0 and differenceY < 0:
        return True
    # if Difference is less than 5% of the sum of the width of both boxes
    if differenceX < (0.05 * halfWidthSumX) and differenceY < (
        0.05 * halfWidthSumY
    ):
        return True
    else:
        return False


def crop(b1, b2, value):
    minx = min(b1[0][0], b2[0][0])
    miny = min(b1[0][1], b2[0][1])
    maxx = max(b1[2][0], b2[2][0])
    maxy = max(b1[2][1], b2[2][1])
    img = value
    h, w = img.shape[:2]

    crop_img = img[
        max(0, int(miny) - 50) : min(h, int(maxy) + 50),
        max(0, int(minx) - 50) : min(w, int(maxx) + 50),
    ]

    # Uncomment for multiprocessing

    # extract_features(crop_img)
    # pid = os.fork()
    # if pid == 0: # Enters only for the child process
    #     sys.exit(0)
    #     os.kill(pid, signal.SIGKILL)

    extract_features(crop_img)


# Time stamp of the previous fraame
def object_detection(image_path):

    image, detections = image_detection(
        image_path, network, class_names, class_colors, args.thresh
    )
    global x_y_values
    x_y_values = []

    for detection in detections:
        label, confidence, boxes = detection  # x1, y1, x2, y2
        if label == "car":
            x1, y1, x2, y2 = darknet.bbox2points(boxes)
            #  = boxes
            temp = []
            temp.append([x1, y1])
            temp.append([x2, y1])
            temp.append([x2, y2])
            temp.append([x1, y2])
            temp.append([x1, abs(x2 - x1)])  # For Half Width Calculation
            temp.append([y1, abs(y2 - y1)])  # For Half Width Calculation

            x_y_values.append(temp)

            image = cv2.rectangle(
                image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2
            )
            # cv2.imwrite("filename.jpg", image)


def start_from_here(path):
    cap = cv2.VideoCapture(path)
    j = 0
    i = 0
    prev_frame = 0
    last_accident = 0
    while True:
        ret, frame = cap.read()
        try:
            if i > 0 and prev_frame == frame:
                break
        except:
            print("End")
            break
        if ret:
            cv2.imshow("Vid frame", frame)

            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
            frame = cv2.resize(frame, (400, 300))  # Frame resize
            # variable i to keep track of the number of frames being sent for evaluation
            if i % 5 == 0:
                j = j + 1
                s = "image.jpg"

                if (time.time() - last_accident) >= 3:
                    # cv2.imwrite(s, frame)
                    object_detection(frame)  # x_y_values are populated
                    checkCollisions(frame)
                else:
                    print("Skipping")
            i = i + 1

            prev_frame = frame
    cv2.destroyAllWindows()


# Load model
model_prediction = tf.keras.models.load_model("classification_models/ANN_model.h5")
model_prediction.predict(np.zeros((1,1920))) # model warmup
model_dense = DenseNet201(weights="imagenet")

# Remove last dense layer
model_new = Model(model_dense.input, model_dense.layers[-2].output)
model_new.predict(np.zeros((1,224,224,3)))
last_accident = 0

cred = credentials.Certificate("firebase_credentials.json")
firebase_admin.initialize_app(
    cred, {"storageBucket": "accident-detection-54513.appspot.com"}
)

args = parser()
check_arguments_errors(args)

random.seed(3)  # deterministic bbox colors
network, class_names, class_colors = darknet.load_network(
    args.config_file, args.data_file, args.weights, batch_size=args.batch_size
)

x_y_values = []
temp = []

# main function
if __name__ == "__main__":
    path = args.path
    start_from_here(path)
