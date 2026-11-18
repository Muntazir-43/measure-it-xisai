
import cv2
import numpy as np
from pathlib import Path

IMAGE_DIR = Path("dataset/images")
MASK_DIR = Path("dataset/masks")

mask_files = sorted(MASK_DIR.glob("*.png"))

print("Total binary masks:", len(mask_files))

errors = []

for mask_path in mask_files:

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        errors.append(f"Unreadable mask: {mask_path.name}")
        continue

    unique_values = np.unique(mask)

    # Check binary pixel values
    if not np.all(np.isin(unique_values, [0, 255])):
        errors.append(
            f"Non-binary values: {mask_path.name}"
        )

    # Check matching original image
    image_path = IMAGE_DIR / (
        mask_path.stem + ".jpg"
    )

    if not image_path.exists():
        errors.append(
            f"Missing original image: {image_path.name}"
        )
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        errors.append(
            f"Unreadable image: {image_path.name}"
        )
        continue

    if image.shape[:2] != mask.shape[:2]:
        errors.append(
            f"Dimension mismatch: {mask_path.name}"
        )

print("\n========== VERIFICATION ==========")

if len(errors) == 0:
    print("SUCCESS: All masks passed basic verification.")
else:
    print("ISSUES FOUND:", len(errors))
    for error in errors:
        print("[ERROR]", error)
