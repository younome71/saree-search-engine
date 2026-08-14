from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


IMAGE_DIR = Path("data/images")

RESULTS = [
    ("000001.webp", 1.0000),
    ("000011.webp", 0.9643),
    ("000010.webp", 0.9641),
    ("000008.webp", 0.9582),
    ("000009.webp", 0.9424),
    ("000012.webp", 0.9413),
    ("000006.webp", 0.9367),
    ("000064.webp", 0.9332),
    ("000065.webp", 0.9296),
    ("000030.webp", 0.9284),
]


def main():

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(16, 8)
    )

    axes = axes.flatten()

    for i, (filename, score) in enumerate(RESULTS):

        image_path = IMAGE_DIR / filename

        image = Image.open(image_path)

        axes[i].imshow(image)
        axes[i].axis("off")

        if i == 0:
            title = f"QUERY\n{filename}"
        else:
            title = (
                f"#{i}  {filename}\n"
                f"Similarity: {score:.4f}"
            )

        axes[i].set_title(title)

    plt.tight_layout()

    output_path = "results_preview.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()