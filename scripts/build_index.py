from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from PIL import Image

from src.embeddings import ImageEmbedder


METADATA_PATH = "data/image_metadata.csv"
IMAGE_DIR = Path("data/images")

INDEX_DIR = Path("index")
INDEX_PATH = INDEX_DIR / "sarees.faiss"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"

BATCH_SIZE = 16


def main():

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Reading metadata...")

    metadata = pd.read_csv(
        METADATA_PATH
    )

    # Only index images that actually exist
    metadata = metadata[
        metadata["image_exists"] == True
    ].copy()

    metadata = metadata.reset_index(
        drop=True
    )

    print(
        f"Images available for indexing: "
        f"{len(metadata)}"
    )

    embedder = ImageEmbedder()

    all_embeddings = []

    total = len(metadata)

    print("\nGenerating embeddings...\n")

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch_metadata = metadata.iloc[
            start:end
        ]

        images = []

        valid_rows = []

        for _, row in batch_metadata.iterrows():

            image_path = (
                IMAGE_DIR /
                row["image_filename"]
            )

            try:

                image = Image.open(
                    image_path
                ).convert("RGB")

                images.append(image)
                valid_rows.append(row)

            except Exception as e:

                print(
                    f"Skipping {image_path}: {e}"
                )

        if not images:
            continue

        embeddings = embedder.encode_images(
            images
        )

        all_embeddings.append(
            embeddings
        )

        print(
            f"[{end}/{total}] "
            f"Processed {len(images)} images"
        )

    # Combine all batches
    embeddings = np.vstack(
        all_embeddings
    ).astype("float32")

    print("\n" + "=" * 60)
    print("EMBEDDINGS COMPLETE")
    print("=" * 60)

    print(
        "Embedding matrix:",
        embeddings.shape
    )

    # Save embeddings separately
    np.save(
        EMBEDDINGS_PATH,
        embeddings
    )

    print(
        f"Saved embeddings to: "
        f"{EMBEDDINGS_PATH}"
    )

    # -------------------------------------------------
    # FAISS
    # -------------------------------------------------

    dimension = embeddings.shape[1]

    # Inner product + normalized vectors
    # = cosine similarity
    base_index = faiss.IndexFlatIP(
        dimension
    )

    # Allows us to associate each vector
    # with our catalogue image_id
    index = faiss.IndexIDMap2(
        base_index
    )

    image_ids = metadata[
        "image_id"
    ].values.astype("int64")

    index.add_with_ids(
        embeddings,
        image_ids
    )

    faiss.write_index(
        index,
        str(INDEX_PATH)
    )

    print(
        f"FAISS index saved to: "
        f"{INDEX_PATH}"
    )

    print(
        f"Vectors in index: "
        f"{index.ntotal}"
    )


if __name__ == "__main__":
    main()