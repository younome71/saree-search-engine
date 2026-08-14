from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"


class ImageEmbedder:

    def __init__(self):
        print(f"Loading model: {MODEL_NAME}")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Using device: {self.device}")

        self.processor = CLIPProcessor.from_pretrained(
            MODEL_NAME
        )

        self.model = CLIPModel.from_pretrained(
            MODEL_NAME
        )

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_images(self, images):
        """
        Encode a batch of PIL images.

        Returns:
            numpy array of shape (batch_size, 512)
        """

        inputs = self.processor(
            images=images,
            return_tensors="pt"
        )

        pixel_values = inputs["pixel_values"].to(
            self.device
        )

        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values
        )

        pooled_output = vision_outputs.pooler_output

        image_features = self.model.visual_projection(
            pooled_output
        )

        # Normalize for cosine similarity
        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True
        )

        return image_features.cpu().numpy()

    def encode_image(self, image):
        return self.encode_images([image])[0]

    def encode_file(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.encode_image(image)


if __name__ == "__main__":

    test_image = Path(
        "data/images/000001.webp"
    )

    embedder = ImageEmbedder()

    embedding = embedder.encode_file(
        test_image
    )

    print("\nEmbedding generated!")
    print("Shape:", embedding.shape)
    print("Norm:", np.linalg.norm(embedding))