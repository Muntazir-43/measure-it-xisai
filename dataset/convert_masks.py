
import cv2
import numpy as np
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_DIR = Path("dataset/masks_exported")
OUTPUT_DIR = Path("dataset/masks")

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# CONVERSION
# ==========================================

mask_files = sorted(INPUT_DIR.glob("*.png"))

print("Total exported masks found:", len(mask_files))

if len(mask_files) == 0:
    raise FileNotFoundError(
        f"No PNG masks found in {INPUT_DIR}"
    )

successful = 0
failed = []

for mask_path in mask_files:

    # Read original CVAT mask in BGR format
    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_COLOR
    )

    if mask is None:
        print(f"[ERROR] Could not read: {mask_path.name}")
        failed.append(mask_path.name)
        continue

    # ======================================
    # DETECT RED SMARTPHONE SEGMENTATION
    # ======================================

    blue = mask[:, :, 0]
    green = mask[:, :, 1]
    red = mask[:, :, 2]

    red_region = (
        (red > 80) &
        (red > green * 1.5) &
        (red > blue * 1.5)
    )

    # ======================================
    # CREATE BINARY MASK
    # ======================================

    binary_mask = np.zeros(
        mask.shape[:2],
        dtype=np.uint8
    )

    binary_mask[red_region] = 255

    # ======================================
    # SAVE MASK
    # ======================================

    output_path = OUTPUT_DIR / mask_path.name

    saved = cv2.imwrite(
        str(output_path),
        binary_mask
    )

    if not saved:
        print(f"[ERROR] Could not save: {mask_path.name}")
        failed.append(mask_path.name)
        continue

    # ======================================
    # VALIDATION
    # ======================================

    unique_values = np.unique(binary_mask)

    if not np.array_equal(
        unique_values,
        np.array([0, 255], dtype=np.uint8)
    ):
        # An entirely empty or full mask needs review
        print(
            f"[WARNING] Unexpected pixel values: "
            f"{mask_path.name}: {unique_values}"
        )

    foreground_pixels = np.count_nonzero(binary_mask)
    total_pixels = binary_mask.size

    foreground_percentage = (
        foreground_pixels / total_pixels
    ) * 100

    print(
        f"[OK] {mask_path.name} | "
        f"Values: {unique_values} | "
        f"Foreground: {foreground_percentage:.2f}%"
    )

    successful += 1

# ==========================================
# FINAL REPORT
# ==========================================

print("\n========== CONVERSION SUMMARY ==========")

print("Total masks found:", len(mask_files))
print("Successfully converted:", successful)
print("Failed:", len(failed))

if failed:
    print("\nFailed files:")
    for name in failed:
        print(name)

print("\nOutput directory:", OUTPUT_DIR)

print("\nBinary mask conversion completed.")
