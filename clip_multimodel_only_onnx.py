#!/usr/bin/env python3
import numpy as np
# noinspection PyPackageRequirements
from PIL import Image
import onnxruntime as ort
import instant_clip_tokenizer

# TEXT_CLASSES = [
#     "boys fighting",
#     "boys playing"
# ]

# noinspection DuplicatedCode
TEXT_CLASSES = [
    "car", "person", "dog", "cat", "orange", "apple",
    "apple with leaves", "rotten apple", "rotten orange",
    "sliced orange", "sliced apple"
]


def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC)
    img = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
    img = ((img - mean) / std).astype(np.float32)
    return np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)


def infer_image(ort_session, image_path, text_tokens_np):
    return ort_session.run(None, {"image": preprocess_image(image_path), "text": text_tokens_np})[0][0]


def print_results(outputs):
    header = f"{'Label':50s} | {'Probability':>10s}"
    sep = "-" * len(header)
    table = "\n".join(f"{label:50s} | {score:10.3f}" for label, score in zip(TEXT_CLASSES, outputs.tolist()))
    top_idx = int(np.argmax(outputs))
    print(f"{header}\n{sep}\n{table}\n{sep}")
    print(f"-> Most probable: '{TEXT_CLASSES[top_idx]}' with probability {outputs[top_idx]:.3f}")


def main():
    ort_session = ort.InferenceSession("clip_multimodal.onnx")
    tokenizer = instant_clip_tokenizer.Tokenizer()
    text_tokens_np = np.array(tokenizer.tokenize_batch(TEXT_CLASSES, context_length=77), dtype=np.int32)
    image_path = "images/rotten-orange.png"
    print(f"Inferencing on image: {image_path}")
    output = infer_image(ort_session, image_path, text_tokens_np)
    print_results(output)


if __name__ == "__main__":
    main()
