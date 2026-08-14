from pathlib import Path
import pandas as pd


CSV_PATH = "data/products.csv"
OUTPUT_PATH = "data/image_metadata.csv"
IMAGE_DIR = Path("data/images")


def main():
    df = pd.read_csv(CSV_PATH)

    # Stable ID corresponding to the original CSV row
    df["image_id"] = range(1, len(df) + 1)

    # Image filename used by our downloader
    df["image_filename"] = df["image_id"].apply(
        lambda x: f"{x:06d}.webp"
    )

    # Check whether the image actually exists
    df["image_exists"] = df["image_filename"].apply(
        lambda x: (IMAGE_DIR / x).exists()
    )

    # Keep only useful columns + original metadata
    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("=" * 50)
    print("METADATA CREATED")
    print("=" * 50)

    print(f"Total catalogue rows : {len(df)}")
    print(f"Images available     : {df['image_exists'].sum()}")
    print(f"Images missing       : {(~df['image_exists']).sum()}")
    print(f"Output               : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()