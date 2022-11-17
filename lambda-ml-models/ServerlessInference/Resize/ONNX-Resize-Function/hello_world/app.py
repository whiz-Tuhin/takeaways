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

def resize(img_from_request):
    resized_img = cv2.resize(img_from_request, (224, 224))
    return resized_img

def decode(image_json):
    decoded_image = base64.b64decode(image_json["image"].encode('utf-8'))
    jpeg_as_np = np.frombuffer(decoded_image, dtype=np.uint8)
    image = cv2.imdecode(jpeg_as_np, flags=1)
    return image

def encode(image):
    retval, buffer = cv2.imencode('.jpg', image)
    encoded_im = base64.b64encode(buffer)
    image_bytes = dict(image=encoded_im.decode('utf-8'))
    return image_bytes


def lambda_handler(event, context): 
    # Get image
    print("Getting Image..")
    post_body = json.loads(event["body"])
    img_base64 = post_body.get('image')
    if img_base64 is None:
        return respond(False, None, "No image parameter received")

    print("Decoding")
    img = decode(post_body)
    print("Resizing")
    resized_image = resize(img)
    print("Encoding")
    resized_encoded = encode(resized_image)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "image": f"{resized_encoded}",
            "prediction": f"None"
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
