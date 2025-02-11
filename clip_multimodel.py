#!/usr/bin/env python3
import os, time
from glob import glob
import numpy as np
# noinspection PyPackageRequirements
from PIL import Image
import onnx, onnxruntime as ort, torch
from torch import nn
from onnxsim import simplify
import clip, instant_clip_tokenizer

# noinspection DuplicatedCode
TEXT_CLASSES = [
    "orange",
    "apple",
    "rotten apple",
    "rotten orange",
    "sliced apple"
]

def preprocess_numpy(path):
    img = Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC)
    img = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
    img = ((img - mean) / std).astype(np.float32)
    return np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

class CLIPMultiModalModel(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.logit_scale = clip_model.logit_scale.exp().detach()

    def forward(self, image, text):
        img_feat = self.visual(image)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        text = text.to(torch.int32)
        x = self.token_embedding(text) + self.positional_embedding
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.ln_final(x)
        eot = torch.argmax(text, dim=-1, keepdim=True)
        sel = torch.gather(x, 1, eot.unsqueeze(-1).expand(-1, 1, x.size(-1))).squeeze(1)
        txt_feat = sel @ self.text_projection
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        return (self.logit_scale * (img_feat @ txt_feat.t())).softmax(dim=-1)

def run_pytorch_inference():
    device, model_name = "cpu", "RN50"
    clip_model, preprocess_fn = clip.load(model_name, device=device, jit=False)
    model = CLIPMultiModalModel(clip_model).to(device).eval()
    image_tensor = preprocess_fn(Image.open("person.jpg")).unsqueeze(0).to(device)
    tokenizer = instant_clip_tokenizer.Tokenizer()
    text_tokens = torch.tensor(
        tokenizer.tokenize_batch(TEXT_CLASSES, context_length=77),
        dtype=torch.int32
    ).to(device)

    with torch.no_grad():
        probs = model(image_tensor, text_tokens)
    for label, p in zip(TEXT_CLASSES, probs[0].tolist()):
        print(f"{label:50s}: {p:.3f}")
    top = probs[0].argmax().item()
    print(f"Most probable: '{TEXT_CLASSES[top]}' with probability {probs[0][top]:.3f}")

    onnx_path = "clip_multimodal.onnx"
    dummy_image = torch.randn(1, 3, 224, 224, device=device)
    dummy_text = torch.tensor(
        tokenizer.tokenize_batch(TEXT_CLASSES, context_length=77),
        dtype=torch.int32
    ).to(device)

    # Export the model with fixed input/output shapes.
    torch.onnx.export(
        model,
        (dummy_image, dummy_text),
        onnx_path,
        input_names=["frame", "text"],  # 'frame' is the image input; 'text' is the text tokens.
        output_names=["probabilities"],
        opset_version=14,
        do_constant_folding=True
    )
    onnx_model, check = simplify(onnx.load(onnx_path), check_n=1)
    if not check:
        raise RuntimeError("ONNX model simplification failed")
    onnx.save(onnx_model, onnx_path)
    print("ONNX model exported & simplified.")

def infer_single_image(ort_session, path, text_tokens_np):
    image_data = preprocess_numpy(path)
    start = time.perf_counter()
    # Use fixed input keys matching the exported model.
    outs = ort_session.run(None, {"frame": image_data, "text": text_tokens_np})[0][0]
    run_time = time.perf_counter() - start
    for label, p in zip(TEXT_CLASSES, outs.tolist()):
        print(f"{label:50s}: {p:.3f}")
    top = int(np.argmax(outs))
    print(f"-> Most probable: '{TEXT_CLASSES[top]}' with probability {outs[top]:.3f}")
    print(f"Inference time: {run_time * 1000:.1f} ms")
    return run_time

def infer_directory(ort_session, images_dir, text_tokens_np):
    files = [f for ext in ("*.png", "*.jpg", "*.jpeg")
             for f in glob(os.path.join(images_dir, ext))]
    if not files:
        print(f"No images found in {images_dir}")
        return
    times = []
    for f in files:
        print(f"\nResults for {os.path.basename(f)}:")
        rt = infer_single_image(ort_session, f, text_tokens_np)
        times.append(rt)
    avg_time = (sum(times) / len(times)) * 1000
    print(f"\nAverage inference time: {avg_time:.1f} ms")

def run_onnxruntime_inference():
    ort_session = ort.InferenceSession("clip_multimodal.onnx")
    tokenizer = instant_clip_tokenizer.Tokenizer()
    # Fixed-size text tokens: shape [5, 77]
    text_tokens_np = np.array(
        tokenizer.tokenize_batch(TEXT_CLASSES, context_length=77),
        dtype=np.int32
    )
    infer_directory(ort_session, "images", text_tokens_np)

if __name__ == "__main__":
    print("=== PyTorch Inference & ONNX Export ===")
    run_pytorch_inference()
    print("\n=== ONNX Runtime Inference ===")
    run_onnxruntime_inference()
