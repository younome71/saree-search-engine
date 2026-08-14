from pathlib import Path

from src.agent import SareeAgent


QUERY_IMAGE = Path(
    "data/images/000713.webp"
)


def main():

    print("=" * 70)
    print("TAILORTALK AGENT TEST")
    print("=" * 70)

    print()
    print(
        f"Query image: {QUERY_IMAGE}"
    )

    agent = SareeAgent(
        image_path=QUERY_IMAGE
    )

    response = agent.chat(
        "Find sarees visually similar to this image."
    )

    print()
    print("=" * 70)
    print("AGENT RESPONSE")
    print("=" * 70)

    print(
        response["message"]
    )

    print()
    print("=" * 70)
    print("SEARCH RESULTS")
    print("=" * 70)

    for i, result in enumerate(
        response["results"],
        start=1
    ):

        print(
            f"{i:2d}. "
            f"{result['image_filename']} | "
            f"{result['score']:.4f} | "
            f"{result['name']}"
        )


if __name__ == "__main__":
    main()