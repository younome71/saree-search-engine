from pathlib import Path

import faiss
import pandas as pd
from PIL import Image

from src.embeddings import ImageEmbedder


INDEX_PATH = "index/sarees.faiss"
METADATA_PATH = "data/image_metadata.csv"


class SareeSearcher:

    def __init__(self):

        print("Loading FAISS index...")

        self.index = faiss.read_index(
            INDEX_PATH
        )

        print(
            f"Loaded {self.index.ntotal} vectors"
        )

        self.metadata = pd.read_csv(
            METADATA_PATH
        )

        self.metadata = self.metadata[
            self.metadata["image_exists"] == True
        ].copy()

        self.metadata = self.metadata.set_index(
            "image_id"
        )

        self.embedder = ImageEmbedder()

    def search(
        self,
        image,
        top_k=5
    ):

        query_embedding = (
            self.embedder.encode_image(
                image
            )
        )

        # FAISS expects shape:
        # (number_of_queries, embedding_dimension)
        query_embedding = (
            query_embedding
            .reshape(1, -1)
            .astype("float32")
        )

        scores, ids = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, image_id in zip(
            scores[0],
            ids[0]
        ):

            if image_id == -1:
                continue

            if image_id not in self.metadata.index:
                continue

            row = self.metadata.loc[
                image_id
            ]

            results.append({
                "image_id": int(image_id),
                "image_filename": row[
                    "image_filename"
                ],
                "name": row["Name"],
                "sku": row["SKU"],
                "score": float(score),
                "image_url": row[
                    "image_url"
                ],
                "website_url": row[
                    "Website Link"
                ],
                "retail_price": row[
                    "Retail Price"
                ],
                "discounted_price": row[
                    "Discounted Price"
                ]
            })

        return results


if __name__ == "__main__":

    searcher = SareeSearcher()

    query_path = Path(
        "data/images/000001.webp"
    )

    query = Image.open(
        query_path
    ).convert("RGB")

    results = searcher.search(
        query,
        top_k=10
    )

    print("\n" + "=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)

    for i, result in enumerate(
        results,
        1
    ):

        print(
            f"{i}. "
            f"{result['image_filename']} | "
            f"{result['score']:.4f} | "
            f"{result['name']}"
        )