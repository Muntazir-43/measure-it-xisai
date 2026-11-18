
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "dataset" / "images"
MASK_DIR = ROOT / "dataset" / "masks"
PRED_DIR = ROOT / "outputs" / "test_predictions"

OUTPUT_DIR = ROOT / "outputs" / "test_comparisons"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (256, 144)


def find_file(folder, stem):
    for extension in [".png", ".jpg", ".jpeg"]:
        candidate = folder / (stem + extension)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find {stem} in {folder}"
    )


def create_overlay(image, mask):
    overlay = image.copy()

    # Highlight foreground in green
    green = np.zeros_like(image)
    green[:, :, 1] = 255

    selected = mask > 127

    overlay[selected] = cv2.addWeighted(
        image, 0.45, green, 0.55, 0
    )[selected]

    return overlay


# Read test image names
split_file = ROOT / "dataset" / "splits" / "test.txt"

with open(split_file, "r") as file:
    image_names = [
        line.strip()
        for line in file
        if line.strip()
    ]


for index, image_name in enumerate(image_names, start=1):

    stem = Path(image_name).stem

    image_path = IMAGE_DIR / image_name
    mask_path = find_file(MASK_DIR, stem)
    prediction_path = PRED_DIR / f"test_{index:03d}.png"

    image = cv2.imread(str(image_path))
    ground_truth = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )
    prediction = cv2.imread(
        str(prediction_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or ground_truth is None or prediction is None:
        raise ValueError(f"Could not read files for {image_name}")

    image = cv2.resize(
        image, IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    ground_truth = cv2.resize(
        ground_truth, IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST
    )

    prediction = cv2.resize(
        prediction, IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST
    )

    gt_overlay = create_overlay(image, ground_truth)
    pred_overlay = create_overlay(image, prediction)

    comparison = np.hstack([
        image,
        gt_overlay,
        pred_overlay
    ])

    output_path = (
        OUTPUT_DIR / f"comparison_{index:03d}.jpg"
    )

    cv2.imwrite(str(output_path), comparison)

    print(f"Saved: {output_path}")

print("\nAll test comparisons created.")
print(f"Output folder: {OUTPUT_DIR}")
