import json
import clip  # pip install git+https://github.com/openai/CLIP.git --user
import onnx
import torch
# noinspection PyPackageRequirements
from PIL import Image
from onnxsim import simplify
from torch import nn

input_width = 224
input_height = 224
model_means = [0.48145466, 0.4578275, 0.40821073]
model_means = [round(255 * m, 2) for m in model_means]  # Convert to 0-255 range
model_stds = [0.26862954, 0.26130258, 0.27577711]
model_stds = [round(255 * s, 2) for s in model_stds]  # Convert to 0-255 range
model_name = "RN50"  # TODO: Replace with the desired clip model version

text_classes = [
     "boys fighting",
     "boys playing"
 ]

text_classes = [
    "orange", "sliced-apple", "", "rotten-apple", "apple"
]

class_names = ";".join([f'{i}:a photo of a {c}' for i, c in enumerate(text_classes)])
scores_output_name = f'scores-{class_names}'
opset_version = 14
onnx_path = "boys.onnx"


class ClipTextualModel(nn.Module):
    """Copied from https://github.com/Lednik7/CLIP-ONNX"""

    def __init__(self, model_in):
        super().__init__()
        self.transformer = model_in.transformer
        self.positional_embedding = model_in.positional_embedding
        self.transformer = model_in.transformer
        self.ln_final = model_in.ln_final
        self.text_projection = model_in.text_projection
        self.token_embedding = model_in.token_embedding

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # needs .float() before .argmax(  ) to work
        x = x[torch.arange(x.shape[0]), text.float().argmax(dim=-1)] @ self.text_projection

        return x


class ClipVisionModel(nn.Module):
    """Adapted from https://github.com/Lednik7/CLIP-ONNX"""

    def __init__(self, model_in, text_classes_in):
        super(ClipVisionModel, self).__init__()

        self.logit_scale = model_in.logit_scale.exp().detach()
        text_model = ClipTextualModel(model_in)
        self.model = model_in.visual
        self.model.eval()

        self.text_features = text_model(clip.tokenize(text_classes_in).cpu())
        self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        self.text_features = self.text_features.detach()

    def forward(self, x):
        image_features = self.model(x)
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = self.logit_scale * (image_features @ self.text_features.t())
        probs = logits_per_image.softmax(dim=-1)
        return probs


# onnx cannot work with cuda
model, preprocess = clip.load(model_name, device="cpu", jit=False)

# Create the model
clip_model = ClipVisionModel(model, text_classes)

# batch first
image = preprocess(Image.open("images/orange.png")).unsqueeze(0).cpu()  # [1, 3, 224, 224]

# Run the model
scores = clip_model(image)
print('PyTorch outputs:', scores)

# Export to ONNX
torch.onnx.export(clip_model, image, onnx_path, opset_version=opset_version,
                  input_names=["image-"], output_names=[scores_output_name])

# Update the ONNX description
import sclblonnx as so

graph = so.graph_from_file(onnx_path)
# Add the model means and standard deviations to the ONNX graph description,
# because that's used by the toolchain to populate some settings.
graph.doc_string = json.dumps({'means': model_means, 'vars': model_stds})
so.graph_to_file(graph, onnx_path, onnx_opset_version=opset_version)

# Simplify the ONNX model
# This step is optional, but it is recommended to reduce the size of the model
# optimize the model for inference
try:
    model = onnx.load(onnx_path)
    model, check = simplify(model, check_n=1)
    assert check, "Couldn't simplify the ONNX model"
    onnx.save_model(model, onnx_path)
except Exception as e:
    print(f'Simplification failed: {e}')
    exit(1)

# Load the ONNX model
import onnxruntime

ort_session = onnxruntime.InferenceSession("boys.onnx")

# Run the model
ort_inputs = {ort_session.get_inputs()[0].name: image.detach().numpy()}
ort_outs = ort_session.run(None, ort_inputs)
print('Labels:', text_classes)
print('ONNX Runtime outputs:', ort_outs)
# Most probable class
print('Most probable class:', text_classes[ort_outs[0].argmax()])

############ test clip on a directory of images ############
from os.path import *
from glob import glob

_here = abspath(dirname(__file__))
images_dir = join(_here, 'images')
images = glob(join(images_dir, '*.png'))

for image_path in images:
    image = preprocess(Image.open(image_path)).unsqueeze(0).cpu()
    ort_inputs = {ort_session.get_inputs()[0].name: image.detach().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    print(f'{basename(image_path)}: {text_classes[ort_outs[0].argmax()]}: {ort_outs[0].max()}')
