from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Smaller lightweight model
MODEL_NAME = "openai/clip-vit-base-patch16"

device = "cpu"

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model.to(device)


def encode_image(image_path):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        features = model.get_image_features(**inputs)

    vector = features[0].cpu().numpy()

    return vector.tolist()


def encode_text(text):

    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():

        features = model.get_text_features(**inputs)

    vector = features[0].cpu().numpy()

    return vector.tolist()