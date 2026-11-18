
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Make the project root available for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.unet import UNet
from training.dataset import SmartphoneDataset


# -----------------------------
# Configuration
# -----------------------------
IMAGE_SIZE = (256, 144)  # width, height
BATCH_SIZE = 2
EPOCHS = 15
LEARNING_RATE = 0.001
PATIENCE = 4

CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

torch.manual_seed(42)
torch.set_num_threads(2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training device: {device}")


# -----------------------------
# Load datasets
# -----------------------------
train_dataset = SmartphoneDataset(
    root_dir=ROOT / "dataset",
    split="train",
    image_size=IMAGE_SIZE
)

val_dataset = SmartphoneDataset(
    root_dir=ROOT / "dataset",
    split="val",
    image_size=IMAGE_SIZE
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# -----------------------------
# Create model
# -----------------------------
model = UNet(
    in_channels=3,
    out_channels=1
).to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# Combined BCE + Dice loss
# -----------------------------
def calculate_loss(logits, masks):
    bce_loss = criterion(logits, masks)

    probabilities = torch.sigmoid(logits)

    intersection = (probabilities * masks).sum(
        dim=(1, 2, 3)
    )

    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + masks.sum(dim=(1, 2, 3))
    )

    dice_score = (
        (2.0 * intersection + 1.0)
        / (denominator + 1.0)
    )

    dice_loss = 1.0 - dice_score.mean()

    return bce_loss + dice_loss


# -----------------------------
# Training and validation
# -----------------------------
best_val_loss = float("inf")
epochs_without_improvement = 0

print("\nStarting U-Net training...\n")

for epoch in range(EPOCHS):

    # Training
    model.train()
    train_loss_total = 0.0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = calculate_loss(logits, masks)

        loss.backward()
        optimizer.step()

        train_loss_total += loss.item()

    average_train_loss = (
        train_loss_total / len(train_loader)
    )

    # Validation
    model.eval()
    val_loss_total = 0.0
    dice_total = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = calculate_loss(logits, masks)

            val_loss_total += loss.item()

            predictions = (
                torch.sigmoid(logits) > 0.5
            ).float()

            intersection = (predictions * masks).sum(
                dim=(1, 2, 3)
            )

            denominator = (
                predictions.sum(dim=(1, 2, 3))
                + masks.sum(dim=(1, 2, 3))
            )

            dice = (
                (2.0 * intersection + 1.0)
                / (denominator + 1.0)
            )

            dice_total += dice.mean().item()

    average_val_loss = val_loss_total / len(val_loader)
    average_dice = dice_total / len(val_loader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {average_train_loss:.4f} | "
        f"Val Loss: {average_val_loss:.4f} | "
        f"Val Dice: {average_dice:.4f}"
    )

    # Save the best model
    if average_val_loss < best_val_loss:

        best_val_loss = average_val_loss
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": average_val_loss,
                "image_size": IMAGE_SIZE,
            },
            BEST_MODEL_PATH
        )

        print("  Best model checkpoint saved.")

    else:
        epochs_without_improvement += 1

    # Early stopping
    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping: validation loss did not improve.")
        break


print("\nTraining finished.")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best model saved at: {BEST_MODEL_PATH}")
