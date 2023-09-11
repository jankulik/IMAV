import os
import cv2


def change_saturation_and_brightness(image, saturation_factor, brightness_factor):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = hsv_image[:, :, 1] * saturation_factor
    hsv_image[:, :, 2] = hsv_image[:, :, 2] * brightness_factor
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


# Source and destination folder paths
source_folder = "data/14-07/raw"
destination_folder = "data/14-07/processed"

# Change saturation and brightness factors
saturation_factor = 1.1
brightness_factor = 3

# Loop over images in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith(".JPG") or filename.endswith(".PNG"):
        # Load image
        image_path = os.path.join(source_folder, filename)
        image = cv2.imread(image_path)

        # Apply saturation and brightness modification
        modified_image = change_saturation_and_brightness(image, saturation_factor, brightness_factor)

        # Save the modified image to the destination folder
        save_path = os.path.join(destination_folder, filename)
        cv2.imwrite(save_path, modified_image)

print("Images processed and saved successfully.")
