
from pathlib import Path
import cv2
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "images"
MASK_DIR = BASE_DIR / "masks"
SPLIT_DIR = BASE_DIR / "splits"

SPLITS = ["train", "val", "test"]

EXPECTED_TOTAL = 70

VALID_MASK_VALUES = {0, 255}


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def load_split(split_name):
    split_file = SPLIT_DIR / f"{split_name}.txt"

    if not split_file.exists():
        print(f"[ERROR] Missing split file: {split_file}")
        return None

    with open(split_file, "r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def check_image_mask(image_name):
    image_path = IMAGE_DIR / image_name
    mask_path = MASK_DIR / (
        Path(image_name).stem + ".png"
    )

    errors = []

    # Check file existence
    if not image_path.exists():
        errors.append("Image missing")
        return errors

    if not mask_path.exists():
        errors.append("Mask missing")
        return errors

    # Read image and mask
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        errors.append("Image cannot be read")
        return errors

    if mask is None:
        errors.append("Mask cannot be read")
        return errors

    # Check dimensions
    if image.shape[:2] != mask.shape[:2]:
        errors.append(
            f"Dimension mismatch: "
            f"Image {image.shape[:2]}, "
            f"Mask {mask.shape[:2]}"
        )

    # Check mask pixel values
    unique_values = set(np.unique(mask).tolist())

    if not unique_values.issubset(VALID_MASK_VALUES):
        errors.append(
            f"Invalid mask values: {sorted(unique_values)}"
        )

    # Check empty masks
    foreground_pixels = np.count_nonzero(mask == 255)

    if foreground_pixels == 0:
        errors.append("Empty mask")

    if foreground_pixels == mask.size:
        errors.append("Entire mask is foreground")

    # Calculate foreground percentage
    foreground_percentage = (
        foreground_pixels / mask.size
    ) * 100

    return errors, unique_values, foreground_percentage


# ==========================================
# MAIN DATASET VALIDATION
# ==========================================

def main():

    print("\n==========================================")
    print("       SMARTPHONE DATASET VALIDATION")
    print("==========================================\n")

    all_split_files = {}
    all_filenames = []

    total_errors = 0
    successful_images = 0

    # --------------------------------------
    # 1. CHECK SPLIT FILES
    # --------------------------------------

    print("STEP 1: CHECKING DATASET SPLITS\n")

    for split in SPLITS:

        filenames = load_split(split)

        if filenames is None:
            total_errors += 1
            continue

        all_split_files[split] = filenames
        all_filenames.extend(filenames)

        print(
            f"[OK] {split.upper()} split: "
            f"{len(filenames)} images"
        )

    # --------------------------------------
    # 2. CHECK SPLIT DUPLICATES
    # --------------------------------------

    print("\nSTEP 2: CHECKING DUPLICATES\n")

    if len(all_filenames) != len(set(all_filenames)):

        duplicates = [
            name for name in set(all_filenames)
            if all_filenames.count(name) > 1
        ]

        print("[ERROR] Duplicate filenames found:")

        for name in duplicates:
            print(f"  {name}")

        total_errors += len(duplicates)

    else:
        print("[OK] No duplicate filenames across splits")

    # --------------------------------------
    # 3. CHECK TOTAL DATASET SIZE
    # --------------------------------------

    print("\nSTEP 3: CHECKING DATASET SIZE\n")

    print(f"Total split entries: {len(all_filenames)}")
    print(f"Expected entries: {EXPECTED_TOTAL}")

    if len(all_filenames) != EXPECTED_TOTAL:
        print("[ERROR] Unexpected dataset size")
        total_errors += 1
    else:
        print("[OK] Dataset size is correct")

    # --------------------------------------
    # 4. VALIDATE IMAGE-MASK PAIRS
    # --------------------------------------

    print("\nSTEP 4: VALIDATING IMAGE-MASK PAIRS\n")

    for image_name in all_filenames:

        result = check_image_mask(image_name)

        if len(result) == 1:
            errors = result[0]
            unique_values = None
            foreground_percentage = None
        else:
            errors, unique_values, foreground_percentage = result

        if errors:

            print(f"[ERROR] {image_name}")

            for error in errors:
                print(f"    - {error}")

            total_errors += len(errors)

        else:

            successful_images += 1

            print(
                f"[OK] {image_name} | "
                f"Mask values: {sorted(unique_values)} | "
                f"Foreground: {foreground_percentage:.2f}%"
            )

    # --------------------------------------
    # 5. CHECK ORPHAN FILES
    # --------------------------------------

    print("\nSTEP 5: CHECKING ORPHAN FILES\n")

    split_names = set(all_filenames)

    if IMAGE_DIR.exists():
        image_files = {
            path.name
            for path in IMAGE_DIR.glob("*.jpg")
        }
    else:
        image_files = set()
        print("[ERROR] Image directory is missing")
        total_errors += 1

    if MASK_DIR.exists():
        mask_files = {
            path.name
            for path in MASK_DIR.glob("*.png")
        }
    else:
        mask_files = set()
        print("[ERROR] Mask directory is missing")
        total_errors += 1

    expected_masks = {
        Path(name).stem + ".png"
        for name in split_names
    }

    orphan_images = image_files - split_names
    orphan_masks = mask_files - expected_masks

    missing_images = split_names - image_files
    missing_masks = expected_masks - mask_files

    for name in sorted(orphan_images):
        print(f"[WARNING] Image not assigned to a split: {name}")

    for name in sorted(orphan_masks):
        print(f"[WARNING] Mask not assigned to a split: {name}")

    for name in sorted(missing_images):
        print(f"[ERROR] Split image missing: {name}")
        total_errors += 1

    for name in sorted(missing_masks):
        print(f"[ERROR] Split mask missing: {name}")
        total_errors += 1

    if not orphan_images and not orphan_masks:
        print("[OK] No orphan images or masks")

    # --------------------------------------
    # 6. FINAL SUMMARY
    # --------------------------------------

    print("\n==========================================")
    print("             FINAL SUMMARY")
    print("==========================================")

    print(f"Expected dataset size: {EXPECTED_TOTAL}")
    print(f"Split entries: {len(all_filenames)}")
    print(f"Successfully validated: {successful_images}")
    print(f"Total errors: {total_errors}")

    if total_errors == 0 and successful_images == EXPECTED_TOTAL:

        print("\nSUCCESS: DATASET VALIDATION PASSED")

        print("Your smartphone dataset is ready")
        print("for the next stage of development.")

    else:

        print("\nVALIDATION FAILED")

        print("Please fix the reported issues before training.")

    print("==========================================\n")


if __name__ == "__main__":
    main()
