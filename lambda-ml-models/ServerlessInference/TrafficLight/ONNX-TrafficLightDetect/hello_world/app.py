import json
import base64
import onnxruntime
import numpy as np
import cv2
from collections import namedtuple


def respond(success, vector=None, reason=None):
    if success:
        if vector == None:
            return {
                "statusCode": 400,
                "body": {"image_vector": None, "error": "Failed to create image vector"}
            }
        return {
            "statusCode": 200,
            "body": {"image_vector": vector}
        }

    else:
        return {
            "statusCode": 400,
            "body": {"image_vector": None, "error": reason}
        }

def preprocess(img_data):
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
         # for each pixel and channel
         # divide the value by 255 to get value between [0, 1]
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data


def run_model(model_path, img):
    ort_sess = onnxruntime.InferenceSession(model_path)
    input_name = ort_sess.get_inputs()[0].name
    outputs = ort_sess.run(None, {input_name: img})
    return outputs

def decode_base64(data):
    img = base64.b64decode(data)
    img = cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img.transpose((2,0,1))
    img = img.reshape(1, 3, 224, 224)
    return img


#load text file as list
def load_labels(path):
    labels = []
    with open(path, 'r') as f:
        for line in f:
            labels.append(line.strip())
    print(f"Labels - {labels}")
    return labels

# map mobilenet outputs to classes
def map_outputs(outputs):
    labels = load_labels('imagenet_classes.txt')
    return labels[np.argmax(outputs)] 

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


def lambda_handler(event, context): 
    # Get image
    print("Getting Image..")
    post_body = json.loads(event["body"])
    img_base64 = post_body.get('image')
    if img_base64 is None:
        return respond(False, None, "No image parameter received")

    print("Decoding")
    img = decode(post_body)
    print("Detecting")
    traffic_color = detect(img)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "image": f"{img_base64}",
            "prediction": f"{traffic_color}"
            # "location": ip.text.replace("\n", "")
        }),
    }

# def lambda_handler(event, context):
#     model_path = "mobilenetv2-12.onnx"
#     # Get image
#     print("Getting Image..")
#     post_body = json.loads(event["body"])
#     img_base64 = post_body.get('image')
#     if img_base64 is None:
#         return respond(False, None, "No image parameter received")

#     print("Decoding and Pre-processing")
#     img = decode_base64(img_base64)
#     img = preprocess(img)

#     print("Running model")
#     outputs = run_model(model_path, img)
#     print(f"Got outputs {outputs}")
    
#     result = map_outputs(outputs)
#     print(f"Result - {result}")

#     return {
#         "statusCode": 200,
#         "body": json.dumps({
#             "image": f"{img_base64}",
#             "prediction": f"{result}"
#             # "location": ip.text.replace("\n", "")
#         }),
#     }
