import torch
import numpy as np
from PIL import Image
import open_clip


MODEL_NAME = "hf-hub:Marqo/marqo-fashionCLIP"


class ImageEmbedder:

    def __init__(self):

        print(
            f"Loading FashionCLIP model: {MODEL_NAME}"
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Device: {self.device}"
        )

        # Official OpenCLIP loading method
        self.model, _, self.preprocess = (
            open_clip.create_model_and_transforms(
                MODEL_NAME
            )
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        print(
            "FashionCLIP loaded successfully."
        )

    def encode_image(self, image):

        if not isinstance(
            image,
            Image.Image
        ):
            image = Image.open(image)

        image = image.convert("RGB")

        image_tensor = self.preprocess(
            image
        ).unsqueeze(0)

        image_tensor = image_tensor.to(
            self.device
        )

        with torch.no_grad():

            features = self.model.encode_image(
                image_tensor,
                normalize=True
            )

        embedding = (
            features
            .cpu()
            .numpy()[0]
        )

        return embedding.astype(
            "float32"
        )

    def encode_images(
        self,
        images,
        batch_size=16
    ):

        all_embeddings = []

        for start in range(
            0,
            len(images),
            batch_size
        ):

            batch = images[
                start:start + batch_size
            ]

            image_tensors = []

            for image in batch:

                if isinstance(
                    image,
                    Image.Image
                ):
                    pil_image = image
                else:
                    pil_image = Image.open(
                        image
                    )

                pil_image = pil_image.convert(
                    "RGB"
                )

                image_tensors.append(
                    self.preprocess(
                        pil_image
                    )
                )

            image_tensor = torch.stack(
                image_tensors
            ).to(self.device)

            with torch.no_grad():

                features = (
                    self.model.encode_image(
                        image_tensor,
                        normalize=True
                    )
                )

            all_embeddings.append(
                features
                .cpu()
                .numpy()
            )

        return np.vstack(
            all_embeddings
        ).astype("float32")


if __name__ == "__main__":

    embedder = ImageEmbedder()

    test_image = Image.open(
        "data/images/000001.webp"
    ).convert("RGB")

    embedding = embedder.encode_image(
        test_image
    )

    print()
    print("=" * 60)
    print("FASHIONCLIP EMBEDDING TEST")
    print("=" * 60)

    print(
        f"Shape : {embedding.shape}"
    )

    print(
        f"Dtype : {embedding.dtype}"
    )

    print(
        f"Norm  : "
        f"{np.linalg.norm(embedding):.6f}"
    )

    print(
        f"First 10 values: "
        f"{embedding[:10]}"
    )