import json
import os

import cv2
import numpy as np
import requests

from Detection import Detection
from Manager import Manager

manager = None


def main_video_device() -> None:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        detection = detect_from_image(frame)
        key = show_image(detection.processed, "Camera")
        upload_data(detection, key)
        if key == 32:
            break


def main_images() -> None:
    image_paths = [image for image in os.listdir("images") if image.endswith(".jpg")]
    i = 0
    while True:
        key, _ = detect_from_path(image_paths[i])
        if key in [ord("n"), ord("s")]:
            i = i + 1 if i < len(image_paths) - 1 else 0
        elif key == 32:
            break


def main_single(image_path) -> Detection:
    _, detection = detect_from_path(image_path)
    return detection


def detect_from_path(image_path: str) -> [int, Detection]:
    image_path = f"images/{image_path}"
    frame = cv2.imread(image_path)
    detection = detect_from_image(frame)
    key = show_image(detection.processed, "Detected")
    upload_data(detection, key)
    return key, detection


def detect_from_image(image: np.ndarray) -> Detection:
    image_blur = cv2.GaussianBlur(image, (manager.gaussian, manager.gaussian), cv2.BORDER_DEFAULT)
    image_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
    ret, image_thresh = cv2.threshold(image_gray, manager.threshold, 255, 0)
    contours, hierarchy = cv2.findContours(image_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if manager.debug:
        show_image(image_thresh, "debug", realtime=True)
    return detect(contours, image)


def detect(contours: np.ndarray, image: np.ndarray) -> Detection:
    detection = Detection(image, None, 0, 0, 0, 0)
    min_contour = 3
    max_contour = 3000

    valid_contours = [contour for contour in contours if
                      max_contour > cv2.contourArea(contour) > min_contour]

    average_area = 0
    if len(valid_contours) != 0:
        average_area = sum([cv2.contourArea(contour) for contour in valid_contours]) / len(
            valid_contours)

    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if max_contour > cv2.contourArea(contour) > min_contour:
            if area < average_area * 0.6:
                color = (255, 0, 0)  # BLUE
                detection.small += 1
            elif area < average_area * 1.2:
                color = (0, 255, 0)  # GREEN
                detection.medium += 1
            elif area < average_area * 4:  # BIG
                color = (0, 0, 255)
                detection.big += 1
            else:
                color = (255, 0, 255)  # PINK
                detection.fly += 1

            draw_contours_centers(image, contour, color)

    message_detection = f"Small={detection.small} Medium={detection.medium} Big={detection.big} Fly={detection.fly}"
    cv2.putText(image, message_detection, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    cv2.putText(image, message_detection, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    if manager.debug:
        message_debug = f"Gaussian={manager.gaussian} Threshold={manager.threshold}"
        cv2.putText(image, message_debug, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    4)
        cv2.putText(image, message_debug, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    detection.processed = image
    return detection


def draw_contours_centers(image: np.ndarray, contour: np.ndarray, color: ()) -> None:
    middle = cv2.moments(contour)
    x = int(middle["m10"] / middle["m00"])
    y = int(middle["m01"] / middle["m00"])
    cv2.circle(image, (x, y), 2, color, 2)


def upload_data(detection, key):
    if key == ord("s"):
        data = {
            "original": detection.original.tolist(),
            "processed": detection.processed.tolist(),
            "small": detection.small,
            "medium": detection.medium,
            "big": detection.big,
            "fly": detection.fly,
        }
        print("Posting image...")
        try:
            requests.post("http://127.0.0.1:5000/add", json=data)
            print("Image uploaded to dashboard")
        except requests.exceptions.RequestException as e:
            print(e)
            print("Oops! It looks like the website is not up")


def show_image(image: np.ndarray, title: str, realtime=False) -> int:
    cv2.imshow(title, image)
    if manager.realtime or realtime:
        return cv2.waitKey(1)
    else:
        return cv2.waitKey(0)


if __name__ == "__main__":
    with open('config.json') as json_file:
        config = json.load(json_file)
    manager = Manager(config)
    # main_video_device()
    main_images()
    # print(main_single("easy_1.jpg"))
