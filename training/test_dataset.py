
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import SmartphoneDataset


ROOT = Path(__file__).resolve().parent.parent / "dataset"

# Load training dataset
train_dataset = SmartphoneDataset(
    root_dir=ROOT,
    split="train",
    augment=True
)

# Load validation dataset
val_dataset = SmartphoneDataset(
    root_dir=ROOT,
    split="val",
    augment=False
)

# Load test dataset
test_dataset = SmartphoneDataset(
    root_dir=ROOT,
    split="test",
    augment=False
)

# Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

images, masks = next(iter(train_loader))

print("\n========== DATASET TEST ==========")

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Testing samples:", len(test_dataset))

print("\nImage tensor shape:", images.shape)
print("Mask tensor shape:", masks.shape)

print("Image minimum:", images.min().item())
print("Image maximum:", images.max().item())

print("Mask unique values:", torch.unique(masks))

print("\nDataset loading successful!")
