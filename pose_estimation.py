import torch
from torchvision import transforms

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import json


def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")
    model = torch.load("models/yolov7-w6-pose.pt", map_location=device)["model"]
    # Put in inference mode
    model.float().eval()

    if torch.cuda.is_available():
        # half() turns predictions into float16 tensors
        # which significantly lowers inference time
        model.half().to(device)
    return model


model = load_model()


def run_inference(url):
    image = cv2.imread(url)  # shape: (480, 640, 3)
    # Resize and pad image
    image = letterbox(image, 960, stride=64, auto=True)[0]  # shape: (768, 960, 3)
    # Apply transforms
    image = transforms.ToTensor()(image)  # torch.Size([3, 768, 960])
    # Turn image into batch
    image = image.unsqueeze(0)  # torch.Size([1, 3, 768, 960])
    output, _ = model(image)  # torch.Size([1, 45900, 57])
    return output, image


def visualize_output(output, image, show=False):
    output = non_max_suppression_kpt(
        output,
        0.25,  # Confidence Threshold
        0.65,  # IoU Threshold
        nc=model.yaml["nc"],  # Number of Classes
        nkpt=model.yaml["nkpt"],  # Number of Keypoints
        kpt_label=True,
    )
    with torch.no_grad():
        output = output_to_keypoint(output)

    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    # body_points = np.zeros((output.shape[0], 17, 2))
    # for idx in range(output.shape[0]):
    #     new_body_points = plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
    #     body_points[idx] = new_body_points

    # # cv2.imwrite("state_images/output.png", nimg)

    # if show:
    #     plt.figure(figsize=(12, 12))
    #     plt.axis("off")
    #     plt.imshow(nimg)
    #     plt.show()

    # return body_points


def calc_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def calc_angle(p1, p2, p3):
    distance12 = calc_distance(p1, p2)
    distance23 = calc_distance(p2, p3)
    distance31 = calc_distance(p3, p1)

    angle = np.arccos((distance31**2 - distance12**2 - distance23**2) / (-2 * distance12 * distance23))
    return angle * 180 / np.pi


with open("state_images/labels.json") as f:
    labels = json.load(f)
directory = "state_images"
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg") and filename != "IMG_1443.jpg":
        print(f"file {filename}")

        file_path = os.path.join(directory, filename)
        output, image = run_inference(file_path)
        body_points = visualize_output(output, image, show=False)

        # angle1 = calc_angle(
        #     [body_points[0, 5, 0], body_points[0, 5, 1]],
        #     [body_points[0, 11, 0], body_points[0, 11, 1]],
        #     [body_points[0, 13, 0], body_points[0, 13, 1]],
        # )

        # angle2 = calc_angle(
        #     [body_points[0, 6, 0], body_points[0, 6, 1]],
        #     [body_points[0, 12, 0], body_points[0, 12, 1]],
        #     [body_points[0, 14, 0], body_points[0, 14, 1]],
        # )

        # min_angle = np.min([angle1, angle2])

        # print(f"file {filename}")
        # print(f"angle1 {angle1}")
        # print(f"angle2 {angle2}")
        # print(f"min angle {min_angle}")
        # print(f"label {labels[filename]}")
        # print("---------------------")
    else:
        continue
