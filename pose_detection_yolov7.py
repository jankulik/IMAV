from ultralytics import YOLO
import cv2
import json
import os
import numpy as np

model = YOLO("models/yolov7-w6-pose.pt")


def calc_distance(point_1, point_2):
    return ((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2) ** 0.5


def calc_angle(point_1, point_2, point_3):
    distance_12 = calc_distance(point_1, point_2)
    distance_23 = calc_distance(point_2, point_3)
    distance_31 = calc_distance(point_3, point_1)

    angle = np.arccos((distance_31**2 - distance_12**2 - distance_23**2) / (-2 * distance_12 * distance_23))
    return angle * 180 / np.pi


def scan_image(image_path, labels=None, save_image=False):
    results = model(image_path)[0]
    detected_objects = results.keypoints.xy.numpy()
    image_name = image_path.split("/")[-1]

    for detected_object in detected_objects:
        if len(detected_object) == 0:
            continue

        angle1 = calc_angle(
            [detected_object[5, 0], detected_object[5, 1]],
            [detected_object[11, 0], detected_object[11, 1]],
            [detected_object[13, 0], detected_object[13, 1]],
        )

        angle2 = calc_angle(
            [detected_object[6, 0], detected_object[6, 1]],
            [detected_object[12, 0], detected_object[12, 1]],
            [detected_object[14, 0], detected_object[14, 1]],
        )

        min_angle = np.min([angle1, angle2])

        print(f"file {image_name}")
        print(f"angle1 {angle1}")
        print(f"angle2 {angle2}")
        print(f"min angle {min_angle}")
        if labels != None:
            print(f"label {labels[image_name]}")
        print("---------------------")

        break

    if save_image:
        image = cv2.imread(image_path)
        for detected_object in detected_objects:
            for point in detected_object:
                image = cv2.circle(image, (int(point[0]), int(point[1])), radius=8, color=(0, 0, 255), thickness=-1)

        cv2.imwrite(os.path.join("results", f"{image_name[:-4]}_detected.png"), image)


def scan_images(directory, save_images=False):
    with open(os.path.join(directory, "labels.json")) as f:
        labels = json.load(f)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            scan_image(image_path, labels, save_image=save_images)
        else:
            continue


scan_images("state_images", save_images=True)
# scan_image("state_images/IMG_1507.jpg", save_image=True)
