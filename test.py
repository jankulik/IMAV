from ultralytics import YOLO
import cv2

model = YOLO("yolov8x-pose-p6.pt")

image_path = "state_images/IMG_1380.jpg"
results = model(image_path)
image = cv2.imread(image_path)

print(results[0])

for r in results:
    arr = r.keypoints.xy.numpy()
    for obj in arr:
        for point in obj:
            print(point)
            image = cv2.circle(image, (int(point[0]), int(point[1])), radius=8, color=(0, 0, 255), thickness=-1)

cv2.imwrite("out.png", image)


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
    if filename.endswith("1606.jpg"):
        print(f"file {filename}")

        file_path = os.path.join(directory, filename)
        output, image = run_inference(file_path)
        body_points = visualize_output(output, image, show=False)
        body_points = None

        print(psutil.virtual_memory().percent)

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
