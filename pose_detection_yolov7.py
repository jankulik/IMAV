import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
import os
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weigths = torch.load("models/yolov7-w6-pose.pt", map_location=device)
model = weigths["model"]
_ = model.float().eval()

if torch.cuda.is_available():
    model.half().to(device)


def scan_image(image_path, labels=None, save_image=False):
    image_name = image_path.split("/")[-1]
    image = cv2.imread(image_path)
    image = letterbox(image, 960, stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))

    if torch.cuda.is_available():
        image = image.half().to(device)
    output, _ = model(image)
    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml["nc"], nkpt=model.yaml["nkpt"], kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    body_points = np.zeros((output.shape[0], 17, 2))
    for idx in range(output.shape[0]):
        new_body_points = plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
        body_points[idx] = new_body_points

    angle1 = calc_angle(
        [body_points[0, 5, 0], body_points[0, 5, 1]],
        [body_points[0, 11, 0], body_points[0, 11, 1]],
        [body_points[0, 13, 0], body_points[0, 13, 1]],
    )

    angle2 = calc_angle(
        [body_points[0, 6, 0], body_points[0, 6, 1]],
        [body_points[0, 12, 0], body_points[0, 12, 1]],
        [body_points[0, 14, 0], body_points[0, 14, 1]],
    )

    min_angle = np.min([angle1, angle2])

    print(f"file {image_name}")
    print(f"angle1 {angle1}")
    print(f"angle2 {angle2}")
    print(f"min angle {min_angle}")
    if labels != None:
        print(f"label {labels[image_name]}")
    print("---------------------")

    if save_image:
        cv2.imwrite(os.path.join("res", f"{image_name[:-4]}_detected.png"), nimg)

    return body_points


def calc_distance(point_1, point_2):
    return ((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2) ** 0.5


def calc_angle(point_1, point_2, point_3):
    distance_12 = calc_distance(point_1, point_2)
    distance_23 = calc_distance(point_2, point_3)
    distance_31 = calc_distance(point_3, point_1)

    angle = np.arccos((distance_31**2 - distance_12**2 - distance_23**2) / (-2 * distance_12 * distance_23))
    return angle * 180 / np.pi


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
