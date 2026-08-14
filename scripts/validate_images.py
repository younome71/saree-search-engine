
from pathlib import Path
from PIL import Image

IMAGE_DIR = Path("data/images")

valid = 0
invalid = []

for image_path in IMAGE_DIR.iterdir():
    if not image_path.is_file():
        continue

    try:
        with Image.open(image_path) as img:
            img.verify()

        valid += 1

    except Exception as e:
        invalid.append((image_path.name, str(e)))


print("=" * 50)
print("IMAGE VALIDATION")
print("=" * 50)

print(f"Total files : {valid + len(invalid)}")
print(f"Valid       : {valid}")
print(f"Invalid     : {len(invalid)}")

if invalid:
    print("\nInvalid images:")

    for filename, error in invalid:
        print(f"- {filename}: {error}")