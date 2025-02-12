import numpy as np, onnxruntime
from PIL import Image

# Constants
W, H = 224, 224
MEANS = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STDS  = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

CLASSES = [
    "orange", "sliced-apple", "rotten-orange", "rotten-apple", "apple"
]

MODEL_PATH = "boys.onnx"

def preprocess(path):
    img = Image.open(path).convert("RGB").resize((W, H), Image.BICUBIC)
    img = np.array(img, np.float32) / 255.0
    img = (img - MEANS) / STDS
    return np.expand_dims(np.transpose(img, (2, 0, 1)), 0)

# Load the ONNX model and run inference on one image.
session = onnxruntime.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
image = preprocess("images/orange.png")
outs = session.run(None, {input_name: image})[0]
pred = int(np.argmax(outs))
print("Predicted:", CLASSES[pred], "with score", np.max(outs))
