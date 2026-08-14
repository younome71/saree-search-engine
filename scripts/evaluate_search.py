from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.search import SareeSearcher


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE_DIR = Path("data/images")
EVALUATION_DIR = Path("evaluation")

NUM_QUERIES = 20
TOP_K = 10
RANDOM_SEED = 42


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading search engine...")

    searcher = SareeSearcher()

    # --------------------------------------------------
    # Select random query images
    # --------------------------------------------------

    metadata = searcher.metadata.reset_index()

    queries = metadata.sample(
        n=NUM_QUERIES,
        random_state=RANDOM_SEED
    )

    print(
        f"\nSelected {len(queries)} query images."
    )

    all_results = []

    # --------------------------------------------------
    # Evaluate each query
    # --------------------------------------------------

    for query_number, (_, query_row) in enumerate(
        queries.iterrows(),
        start=1
    ):

        query_image_id = int(
            query_row["image_id"]
        )

        query_filename = query_row[
            "image_filename"
        ]

        query_path = (
            IMAGE_DIR /
            query_filename
        )

        print(
            f"\n[{query_number}/{NUM_QUERIES}] "
            f"Query: {query_filename}"
        )

        query_image = Image.open(
            query_path
        ).convert("RGB")

        # Retrieve extra result because
        # the query itself will normally be #1.
        results = searcher.search(
            query_image,
            top_k=TOP_K + 1
        )

        # Remove query image itself
        results = [
            result
            for result in results
            if result["image_id"] != query_image_id
        ]

        results = results[:TOP_K]

        # --------------------------------------------------
        # Store results
        # --------------------------------------------------

        for rank, result in enumerate(
            results,
            start=1
        ):

            all_results.append({
                "query_number": query_number,
                "query_image_id": query_image_id,
                "query_filename": query_filename,
                "rank": rank,
                "result_image_id": result[
                    "image_id"
                ],
                "result_filename": result[
                    "image_filename"
                ],
                "score": result["score"],
                "name": result["name"],
                "sku": result["sku"],
            })

        # --------------------------------------------------
        # Create contact sheet
        # --------------------------------------------------

        fig, axes = plt.subplots(
            3,
            4,
            figsize=(12, 12)
        )

        axes = axes.flatten()

        # Query image
        query_display = Image.open(
            query_path
        ).convert("RGB")

        axes[0].imshow(
            query_display
        )

        axes[0].set_title(
            f"QUERY\n{query_filename}",
            fontsize=10,
            fontweight="bold"
        )

        axes[0].axis("off")

        # Results
        for i, result in enumerate(
            results[:10],
            start=1
        ):

            result_path = (
                IMAGE_DIR /
                result["image_filename"]
            )

            image = Image.open(
                result_path
            ).convert("RGB")

            axes[i].imshow(image)

            axes[i].set_title(
                f"#{i} "
                f"{result['image_filename']}\n"
                f"Score: {result['score']:.4f}",
                fontsize=9
            )

            axes[i].axis("off")

        # Hide unused axes
        for i in range(
            len(results) + 1,
            len(axes)
        ):
            axes[i].axis("off")

        fig.suptitle(
            f"Query {query_number}: "
            f"{query_row['Name']}",
            fontsize=14,
            fontweight="bold"
        )

        plt.tight_layout()

        output_path = (
            EVALUATION_DIR /
            f"query_{query_number:02d}.png"
        )

        plt.savefig(
            output_path,
            dpi=120,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(
            f"Saved: {output_path}"
        )

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    csv_path = (
        EVALUATION_DIR /
        "results.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print(
        f"Queries evaluated : {NUM_QUERIES}"
    )

    print(
        f"Results per query : {TOP_K}"
    )

    print(
        f"Total results     : {len(results_df)}"
    )

    print(
        f"Average score     : "
        f"{results_df['score'].mean():.4f}"
    )

    print(
        f"Median score      : "
        f"{results_df['score'].median():.4f}"
    )

    print(
        f"\nResults saved to: {csv_path}"
    )


if __name__ == "__main__":
    main()