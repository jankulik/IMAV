from ultralytics import YOLO
import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.neighbors import KNeighborsClassifier

model = YOLO("models/yolov8x-pose.pt")


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

        shoulder_distance = calc_distance(
            [detected_object[5, 0], detected_object[5, 1]], [detected_object[6, 0], detected_object[6, 1]]
        )
        torso_distance = np.mean(
            [
                calc_distance(
                    [detected_object[5, 0], detected_object[5, 1]], [detected_object[11, 0], detected_object[11, 1]]
                ),
                calc_distance(
                    [detected_object[6, 0], detected_object[6, 1]], [detected_object[12, 0], detected_object[12, 1]]
                ),
            ]
        )
        leg_distance = np.mean(
            [
                calc_distance(
                    [detected_object[11, 0], detected_object[11, 1]], [detected_object[15, 0], detected_object[15, 1]]
                ),
                calc_distance(
                    [detected_object[12, 0], detected_object[12, 1]], [detected_object[16, 0], detected_object[16, 1]]
                ),
            ]
        )

        torso_ratio = torso_distance / shoulder_distance
        leg_ratio = leg_distance / shoulder_distance

        print(f"File name: {image_name}")
        print(f"Torso distance ratio: {torso_ratio}")
        print(f"Leg distance ratio: {leg_ratio}")
        if labels != None:
            print(f"Label: {labels[image_name]}")
        print("---------------------------------")

        if labels[image_name] == "lying":
            color = "red"
            marker = "o"
        elif labels[image_name] == "sitting":
            color = "green"
            marker = "^"
        elif labels[image_name] == "standing":
            color = "blue"
            marker = "D"
        plt.plot(torso_ratio, leg_ratio, color=color, marker=marker)

        break

    if save_image:
        image = cv2.imread(image_path)
        for detected_object in detected_objects:
            for point in detected_object:
                image = cv2.circle(image, (int(point[0]), int(point[1])), radius=8, color=(0, 0, 255), thickness=-1)

        cv2.imwrite(os.path.join("state_results", f"{image_name.split('.')[-2]}_detected.png"), image)


def scan_images(directory, save_images=False):
    with open(os.path.join(directory, "labels.json")) as f:
        labels = json.load(f)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpeg"):
            image_path = os.path.join(directory, filename)
            scan_image(image_path, labels, save_image=save_images)
        else:
            continue

    red_circle = mlines.Line2D([], [], color="red", marker="o", linestyle="None", markersize=6, label="lying")
    green_triangle = mlines.Line2D([], [], color="green", marker="^", linestyle="None", markersize=6, label="sitting")
    blue_diamond = mlines.Line2D([], [], color="blue", marker="D", linestyle="None", markersize=6, label="standing")
    plt.legend(handles=[red_circle, green_triangle, blue_diamond])
    plt.title("Torso and leg distance ratios for different classes")
    plt.xlabel("Torso distance ratio")
    plt.ylabel("Leg distance ratio")
    plt.savefig("pose_knn.png")


scan_images("data/poses", save_images=True)
# scan_image("data/poses/IMG_1507.jpg", save_image=True)
