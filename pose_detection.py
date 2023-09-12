from ultralytics import YOLO
import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.neighbors import KNeighborsClassifier
import pickle

model = YOLO("models/yolov8x-pose.pt")


class DetectedObject:
    def __init__(self, image_name, keypoints, label):
        self.image_name = image_name
        self.keypoints = keypoints
        self.label = label

        shoulder_distance = self.calc_distance([keypoints[5, 0], keypoints[5, 1]], [keypoints[6, 0], keypoints[6, 1]])
        torso_distance = np.mean(
            [
                self.calc_distance([keypoints[5, 0], keypoints[5, 1]], [keypoints[11, 0], keypoints[11, 1]]),
                self.calc_distance([keypoints[6, 0], keypoints[6, 1]], [keypoints[12, 0], keypoints[12, 1]]),
            ]
        )
        leg_distance = np.mean(
            [
                self.calc_distance([keypoints[11, 0], keypoints[11, 1]], [keypoints[15, 0], keypoints[15, 1]]),
                self.calc_distance([keypoints[12, 0], keypoints[12, 1]], [keypoints[16, 0], keypoints[16, 1]]),
            ]
        )

        self.torso_ratio = torso_distance / shoulder_distance
        self.leg_ratio = leg_distance / shoulder_distance

    def calc_distance(self, point_1, point_2):
        return ((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2) ** 0.5

    def calc_angle(self, point_1, point_2, point_3):
        distance_12 = self.calc_distance(point_1, point_2)
        distance_23 = self.calc_distance(point_2, point_3)
        distance_31 = self.calc_distance(point_3, point_1)

        angle = np.arccos((distance_31**2 - distance_12**2 - distance_23**2) / (-2 * distance_12 * distance_23))
        return angle * 180 / np.pi


def scan_image(image_path, label=None, save_image=False):
    results = model.predict(image_path, imgsz=1824)[0]
    keypoints_sets = results.keypoints.xy.numpy()
    image_name = image_path.split("/")[-1]

    detected_objects = []
    for keypoints in keypoints_sets:
        if len(keypoints) == 0:
            continue

        detected_object = DetectedObject(image_name, keypoints, label)
        detected_objects.append(detected_object)

        print("-----------------------------------------------")
        print(f"File name: {image_name}")
        print(f"Torso distance ratio: {detected_object.torso_ratio}")
        print(f"Leg distance ratio: {detected_object.leg_ratio}")
        if label != None:
            print(f"Label: {label}")
            if label == "lying":
                color = "red"
                marker = "o"
            elif label == "sitting":
                color = "green"
                marker = "^"
            elif label == "standing":
                color = "blue"
                marker = "D"
            plt.plot(detected_object.torso_ratio, detected_object.leg_ratio, color=color, marker=marker)

    if save_image:
        image = cv2.imread(image_path)
        for keypoints in keypoints_sets:
            for keypoint in keypoints:
                image = cv2.circle(
                    image, (int(keypoint[0]), int(keypoint[1])), radius=8, color=(0, 0, 255), thickness=-1
                )

        cv2.imwrite(os.path.join("results", "poses", f"{image_name.split('.')[-2]}_detected.png"), image)

    return detected_objects


def train(directory, save_images=False):
    with open(os.path.join(directory, "labels.json")) as f:
        labels = json.load(f)

    detected_objects = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename in labels:
            image_path = os.path.join(directory, filename)
            label = labels[filename]
            detected_objects += scan_image(image_path, label, save_image=save_images)
        else:
            continue

    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    x_data = []
    y_data = []
    for detected_object in detected_objects:
        x_data.append([detected_object.torso_ratio, detected_object.leg_ratio])
        y_data.append(detected_object.label)
    knn_classifier.fit(x_data, y_data)

    knn_file = open("models/knn_file", "wb")
    pickle.dump(knn_classifier, knn_file)
    knn_file.close()

    red_circle = mlines.Line2D([], [], color="red", marker="o", linestyle="None", markersize=6, label="lying")
    green_triangle = mlines.Line2D([], [], color="green", marker="^", linestyle="None", markersize=6, label="sitting")
    blue_diamond = mlines.Line2D([], [], color="blue", marker="D", linestyle="None", markersize=6, label="standing")
    plt.legend(handles=[red_circle, green_triangle, blue_diamond])
    plt.title("Torso and leg distance ratios for different classes")
    plt.xlabel("Torso distance ratio")
    plt.ylabel("Leg distance ratio")
    plt.savefig("pose_knn.png")


def run_inference(image_path, save_image=False):
    detected_objects = scan_image(image_path)
    knn_classifier = pickle.load(open("models/knn_file", "rb"))

    image = cv2.imread(image_path)
    for detected_object in detected_objects:
        result = knn_classifier.predict([[detected_object.torso_ratio, detected_object.leg_ratio]])
        print(result)
        for keypoint in detected_object.keypoints:
            image = cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), radius=8, color=(0, 0, 255), thickness=-1)

        image = cv2.putText(
            image,
            result[0],
            org=(int(np.min(detected_object.keypoints[:, 0])), int(np.min(detected_object.keypoints[:, 1]) - 30)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.5,
            color=(0, 0, 255),
            thickness=4,
        )

    cv2.imwrite("inference.png", image)


# train("data/poses", save_images=True)
# run_inference("data/poses/2023_0912_061349_175.JPG", save_image=True)
# scan_image("data/poses/1694459588232.jpeg", save_image=True)


def analyse_video():
    cap = cv2.VideoCapture("rtsp://192.168.43.1:8554/fpv_stream")
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            cv2.imwrite("captured_frame.png", frame)
            break

    cap.release()
    print("kutas")
