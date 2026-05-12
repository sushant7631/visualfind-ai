from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Detect Apple Silicon GPU (M2)
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")

# Load CLIP model
model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

# Move model to GPU
model.to(device)


def encode_image(image_path):
    """
    Convert image into vector
    """

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():

        image_features = model.get_image_features(
            **inputs
        )

    # Normalize vector
    image_features = image_features / image_features.norm(
        dim=-1,
        keepdim=True
    )

    return image_features[0].cpu().numpy().tolist()


def encode_text(text):
    """
    Convert text into vector
    """

    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():

        text_features = model.get_text_features(
            **inputs
        )

    # Normalize vector
    text_features = text_features / text_features.norm(
        dim=-1,
        keepdim=True
    )

    return text_features[0].cpu().numpy().tolist()
def encode_uploaded_image(image):
    """
    Convert uploaded image into vector
    """

    image = image.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():

        image_features = model.get_image_features(
            **inputs
        )

    image_features = image_features / image_features.norm(
        dim=-1,
        keepdim=True
    )

    return image_features[0].cpu().numpy().tolist()