
import cv2
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "dataset" / "images"
MASK_DIR = PROJECT_ROOT / "dataset" / "masks"

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "visualizations"
OVERLAY_DIR = OUTPUT_DIR / "overlays"
COMPARISON_DIR = OUTPUT_DIR / "comparisons"

# Red overlay color (BGR format)
OVERLAY_COLOR = (0, 0, 255)

# Transparency of the red overlay
ALPHA = 0.40

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# FIND IMAGE FILES
# ============================================================

image_files = sorted(IMAGE_DIR.glob("smartphone_*.jpg"))

print(f"Total smartphone images found: {len(image_files)}")

if not image_files:
    raise FileNotFoundError(
        f"No images found in {IMAGE_DIR}"
    )

successful = 0
failed = 0

# ============================================================
# PROCESS EACH IMAGE AND MASK
# ============================================================

for image_path in image_files:

    filename = image_path.stem
    mask_path = MASK_DIR / f"{filename}.png"

    # --------------------------------------------------------
    # CHECK MASK EXISTS
    # --------------------------------------------------------

    if not mask_path.exists():
        print(f"[FAILED] Missing mask: {mask_path.name}")
        failed += 1
        continue

    # --------------------------------------------------------
    # LOAD IMAGE AND MASK
    # --------------------------------------------------------

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"[FAILED] Could not read {filename}")
        failed += 1
        continue

    # --------------------------------------------------------
    # CHECK IMAGE AND MASK DIMENSIONS
    # --------------------------------------------------------

    if image.shape[:2] != mask.shape[:2]:
        print(
            f"[FAILED] Dimension mismatch: {filename} | "
            f"Image: {image.shape[:2]} | "
            f"Mask: {mask.shape[:2]}"
        )
        failed += 1
        continue

    # --------------------------------------------------------
    # ENSURE BINARY MASK
    # --------------------------------------------------------

    binary_mask = np.where(mask > 127, 255, 0).astype(np.uint8)

    unique_values = np.unique(binary_mask)

    # --------------------------------------------------------
    # CREATE COLORED OVERLAY
    # --------------------------------------------------------

    overlay = image.copy()

    colored_mask = np.zeros_like(image)
    colored_mask[:, :] = OVERLAY_COLOR

    foreground = binary_mask == 255

    overlay[foreground] = cv2.addWeighted(
        image[foreground],
        1.0 - ALPHA,
        colored_mask[foreground],
        ALPHA,
        0
    )

    # --------------------------------------------------------
    # CREATE COMPARISON IMAGE
    # --------------------------------------------------------

    # Resize for display only
    display_width = 800
    scale = display_width / image.shape[1]

    display_height = int(image.shape[0] * scale)

    original_display = cv2.resize(
        image,
        (display_width, display_height)
    )

    mask_display = cv2.resize(
        binary_mask,
        (display_width, display_height),
        interpolation=cv2.INTER_NEAREST
    )

    overlay_display = cv2.resize(
        overlay,
        (display_width, display_height)
    )

    # Convert grayscale mask to 3-channel image
    mask_display = cv2.cvtColor(
        mask_display,
        cv2.COLOR_GRAY2BGR
    )

    comparison = np.hstack((
        original_display,
        mask_display,
        overlay_display
    ))

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    overlay_output = OVERLAY_DIR / f"{filename}_overlay.jpg"

    comparison_output = (
        COMPARISON_DIR / f"{filename}_comparison.jpg"
    )

    cv2.imwrite(str(overlay_output), overlay)
    cv2.imwrite(str(comparison_output), comparison)

    successful += 1

    foreground_percentage = (
        np.count_nonzero(binary_mask) / binary_mask.size
    ) * 100

    print(
        f"[OK] {filename} | "
        f"Mask values: {unique_values.tolist()} | "
        f"Foreground: {foreground_percentage:.2f}%"
    )

# ============================================================
# SUMMARY
# ============================================================

print("\n========== VISUALIZATION SUMMARY ==========")

print(f"Total images found: {len(image_files)}")
print(f"Successfully processed: {successful}")
print(f"Failed: {failed}")

print(f"\nOverlay images: {OVERLAY_DIR}")
print(f"Comparison images: {COMPARISON_DIR}")

print("\nVisualization generation completed.")
