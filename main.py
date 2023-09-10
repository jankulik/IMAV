import cv2
import numpy as np
from osgeo import osr, gdal

gdal.PushErrorHandler("CPLQuietErrorHandler")


def get_hsv_range(image, lower_color, upper_color):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_color, upper_color)
    return mask


def draw_shortest_path(img, image_path, starting_location, locations, iteration_number):
    print(
        f"Location of marker {iteration_number + 1}: {get_pixel_latlon(image_path, starting_location[0], starting_location[1])}"
    )

    if locations.shape[0] == 0:
        return

    distances_from_starting_location = np.linalg.norm(locations - starting_location, axis=1)
    next_location_index = np.argmin(distances_from_starting_location)
    next_location = locations[next_location_index]
    cv2.line(img, tuple(starting_location), tuple(next_location), (0, 255, 0), 5)
    new_locations = np.delete(locations, next_location_index, axis=0)

    draw_shortest_path(
        img,
        image_path,
        next_location,
        new_locations,
        iteration_number=iteration_number + 1,
    )


def get_pixel_latlon(image_path, pixel_x, pixel_y):
    ds = gdal.Open(image_path)
    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(ds.GetProjectionRef())

    wgs84_wkt = """
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]]"""
    new_cs = osr.SpatialReference()
    new_cs.ImportFromWkt(wgs84_wkt)

    transform = osr.CoordinateTransformation(old_cs, new_cs)

    gt = ds.GetGeoTransform()
    pixel_x_transformed = gt[0] + pixel_x * gt[1]
    pixel_y_transformed = gt[3] + pixel_y * gt[5]

    lat, lon, _ = transform.TransformPoint(pixel_x_transformed, pixel_y_transformed)
    return lat, lon


def find_path(image_path, num_points=4, show=False):
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    hikers_location = (width, height)

    lower_blue = np.array([100, 150, 150])
    upper_blue = np.array([130, 255, 255])

    blue_mask = get_hsv_range(img, lower_blue, upper_blue)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:num_points]
    centroids = np.empty((0, 2), dtype=int)

    for index, contour in enumerate(contours):
        epsilon = 0.02 * cv2.arcLength(contour, True)
        points = cv2.approxPolyDP(contour, epsilon, True)
        points_squeezed = np.squeeze(points)

        centroid = np.round(np.mean(points_squeezed, axis=0)).astype(int)
        centroids = np.vstack((centroids, centroid))

        cv2.drawMarker(img, tuple(centroid), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.drawContours(img, [points], -1, (0, 255, 0), 3)

    distances_from_hikers = np.linalg.norm(centroids - hikers_location, axis=1)
    starting_location_index = np.argmin(distances_from_hikers)
    starting_location = centroids[starting_location_index]
    draw_shortest_path(
        img,
        image_path,
        starting_location,
        np.delete(centroids, starting_location_index, axis=0),
        iteration_number=0,
    )

    cv2.imwrite("data/ortophoto_trail.png", img)

    if show:
        cv2.imshow("Detected Blue Squares", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


find_path("data/orthophoto.tif", show=False)
