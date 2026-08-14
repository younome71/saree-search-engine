import io
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from PIL import Image


CSV_PATH = "data/products.csv"
OUTPUT_DIR = Path("data/images")

MAX_WORKERS = 12
TIMEOUT = 20

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0 Safari/537.36"
    )
}


def download_image(row):
    row_number = row["row_number"]
    url = row["image_url"]

    filename = f"{row_number:06d}.webp"
    output_path = OUTPUT_DIR / filename

    # Don't download if already present
    if output_path.exists():
        return row_number, "exists"

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=TIMEOUT
        )

        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content))
        image = image.convert("RGB")

        image.save(
            output_path,
            "WEBP",
            quality=95
        )

        return row_number, "success"

    except Exception as e:
        return row_number, f"failed: {e}"


def main():

    print("Reading CSV...")

    df = pd.read_csv(CSV_PATH)

    # Create a stable ID for every CSV row
    df["row_number"] = range(1, len(df) + 1)

    print(f"Total rows: {len(df)}")
    print(f"Unique image URLs: {df['image_url'].nunique()}")
    print(f"Unique SKUs: {df['SKU'].nunique()}")

    print("\nStarting downloads...\n")

    success = 0
    exists = 0
    failed = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = [
            executor.submit(download_image, row)
            for _, row in df.iterrows()
        ]

        for i, future in enumerate(as_completed(futures), 1):

            row_number, status = future.result()

            if status == "success":
                success += 1
                print(f"[{i}/{len(futures)}] ✓ {row_number:06d}")

            elif status == "exists":
                exists += 1
                print(f"[{i}/{len(futures)}] → {row_number:06d} already exists")

            else:
                failed.append({
                    "row_number": row_number,
                    "image_url": df.loc[
                        df["row_number"] == row_number,
                        "image_url"
                    ].iloc[0],
                    "error": status
                })

                print(f"[{i}/{len(futures)}] ✗ {row_number:06d} - {status}")

    # Save failed downloads
    if failed:

        failed_df = pd.DataFrame(failed)

        failed_path = "data/failed_downloads.csv"
        failed_df.to_csv(
            failed_path,
            index=False
        )

        print(f"\nFailed downloads: {len(failed)}")
        print(f"Saved to: {failed_path}")

    print("\n" + "=" * 50)
    print("DOWNLOAD COMPLETE")
    print("=" * 50)

    print(f"Total rows     : {len(df)}")
    print(f"Downloaded     : {success}")
    print(f"Already existed: {exists}")
    print(f"Failed         : {len(failed)}")
    print(f"Images folder  : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()