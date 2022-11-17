import os
import cv2
import numpy as np
import base64
import json
from python.src.utils.classes.commons.serwo_objects import SerWOObject
def detect(img_from_request):
    # Important simplyfying assumption. Exactly one traffic light is in the view of the camera.
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = img_from_request
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # color range
    lower_red1 = np.array([0,100,100])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([160,100,100])
    upper_red2 = np.array([180,255,255])
    lower_green = np.array([40,50,50])
    upper_green = np.array([90,255,255])
    lower_yellow = np.array([15,150,150])
    upper_yellow = np.array([35,255,255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    maskg = cv2.inRange(hsv, lower_green, upper_green)
    masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
    maskr = cv2.add(mask1, mask2)
    # hough circle detect
    r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80,
                                 param1=50, param2=10, minRadius=0, maxRadius=30)
    g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 60,
                                 param1=50, param2=10, minRadius=0, maxRadius=30)
    y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, 30,
                                 param1=50, param2=5, minRadius=0, maxRadius=30)
    # traffic light detect
    r = 5
    bound = 4.0 / 10
    if r_circles is not None:
        return "red"
    if g_circles is not None:
        return "green"
    if y_circles is not None:
        return "yellow"
        
def decode(image_json):
    decoded_image = base64.b64decode(image_json["image"].encode('utf-8'))
    jpeg_as_np = np.frombuffer(decoded_image, dtype=np.uint8)
    image = cv2.imdecode(jpeg_as_np, flags=1)
    return image


def function(serwoObject) -> SerWOObject:
    try:
        image_json = json.loads(serwoObject.get_body())
        image = decode(image_json)
        traffic_color = detect(image)
        ret_val = {"color":traffic_color}
        return SerWOObject(body=ret_val)
    except Exception as e:
        return SerWOObject(error=True)