
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.unet import UNet
from training.dataset import SmartphoneDataset


# Configuration
IMAGE_SIZE = (256, 144)
BATCH_SIZE = 1

CHECKPOINT_PATH = ROOT / "checkpoints" / "best_model.pth"
OUTPUT_DIR = ROOT / "outputs" / "test_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Evaluation device: {device}")

# Load test dataset
test_dataset = SmartphoneDataset(
    root_dir=ROOT / "dataset",
    split="test",
    image_size=IMAGE_SIZE
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# Load trained model
model = UNet(
    in_channels=3,
    out_channels=1
).to(device)

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Loaded checkpoint from epoch: {checkpoint['epoch']}")
print(f"Checkpoint validation loss: {checkpoint['val_loss']:.4f}")

# Evaluation
total_tp = 0
total_fp = 0
total_fn = 0
total_dice = 0.0
total_iou = 0.0

print("\n========== TEST RESULTS ==========")

with torch.no_grad():

    for index, (images, masks) in enumerate(test_loader):

        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        predictions = (
            torch.sigmoid(logits) > 0.5
        ).float()

        # Calculate pixel counts
        tp = (predictions * masks).sum().item()

        fp = (
            predictions * (1 - masks)
        ).sum().item()

        fn = (
            (1 - predictions) * masks
        ).sum().item()

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Per-image Dice and IoU
        dice = (2 * tp + 1) / (2 * tp + fp + fn + 1)
        iou = (tp + 1) / (tp + fp + fn + 1)

        total_dice += dice
        total_iou += iou

        # Save predicted binary mask
        prediction_array = (
            predictions[0, 0].cpu().numpy() * 255
        ).astype(np.uint8)

        output_path = OUTPUT_DIR / f"test_{index + 1:03d}.png"

        cv2.imwrite(str(output_path), prediction_array)

        print(
            f"Test image {index + 1}: "
            f"Dice = {dice:.4f}, IoU = {iou:.4f}"
        )

# Overall pixel-level metrics
overall_dice = (
    (2 * total_tp + 1)
    / (2 * total_tp + total_fp + total_fn + 1)
)

overall_iou = (
    (total_tp + 1)
    / (total_tp + total_fp + total_fn + 1)
)

precision = total_tp / (total_tp + total_fp + 1e-8)
recall = total_tp / (total_tp + total_fn + 1e-8)

mean_dice = total_dice / len(test_dataset)
mean_iou = total_iou / len(test_dataset)

print("\n========== FINAL METRICS ==========")
print(f"Mean per-image Dice: {mean_dice:.4f}")
print(f"Mean per-image IoU: {mean_iou:.4f}")
print(f"Overall pixel Dice: {overall_dice:.4f}")
print(f"Overall pixel IoU: {overall_iou:.4f}")
print(f"Pixel Precision: {precision:.4f}")
print(f"Pixel Recall: {recall:.4f}")

print(f"\nPredicted masks saved in: {OUTPUT_DIR}")
print("\nTest evaluation completed.")
