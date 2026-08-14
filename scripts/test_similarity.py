import numpy as np
from pathlib import Path
from PIL import Image

from src.embeddings import ImageEmbedder


IMAGE_DIR = Path("data/images")

TOP_K = 10


def cosine_similarity(query, embeddings):
    return embeddings @ query


def main():

    embedder = ImageEmbedder()

    image_paths = sorted(
        IMAGE_DIR.glob("*.webp")
    )

    print(f"Found {len(image_paths)} images.")

    # Use the first image as our query
    query_path = image_paths[0]

    print(f"\nQuery image: {query_path}")

    query_embedding = embedder.encode_file(
        query_path
    )

    results = []

    # Initial experiment:
    # encode a small sample rather than all 1069 images
    sample_paths = image_paths[:100]

    print(
        f"\nEncoding {len(sample_paths)} images..."
    )

    for i, path in enumerate(sample_paths):

        embedding = embedder.encode_file(path)

        score = float(
            np.dot(query_embedding, embedding)
        )

        results.append(
            (score, path)
        )

        print(
            f"[{i + 1}/{len(sample_paths)}] "
            f"{path.name} → {score:.4f}"
        )

    results.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    print("\n" + "=" * 60)
    print("TOP MATCHES")
    print("=" * 60)

    for rank, (score, path) in enumerate(
        results[:TOP_K],
        start=1
    ):
        print(
            f"{rank:2d}. "
            f"{path.name} "
            f"score={score:.4f}"
        )


if __name__ == "__main__":
    main()