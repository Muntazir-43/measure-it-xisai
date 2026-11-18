
import cv2
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import Dataset


class SmartphoneDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="train",
        image_size=(256, 144),
        augment=False
    ):

        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.mask_dir = self.root_dir / "masks"
        self.split_file = (
            self.root_dir / "splits" / f"{split}.txt"
        )

        self.image_size = image_size
        self.augment = augment

        if not self.split_file.exists():
            raise FileNotFoundError(self.split_file)

        with open(self.split_file, "r", encoding="utf-8") as f:
            self.image_names = [
                line.strip()
                for line in f
                if line.strip()
            ]

        print(
            f"Loaded {len(self.image_names)} "
            f"images for {split} split."
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):

        image_name = self.image_names[index]

        image_path = self.image_dir / image_name

        mask_path = (
            self.mask_dir /
            (Path(image_name).stem + ".png")
        )

        # Load RGB image
        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(f"Cannot read image: {image_path}")

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Load grayscale mask
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            raise RuntimeError(f"Cannot read mask: {mask_path}")

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image/mask dimensions differ: {image_name}"
            )

        # Resize image and mask
        image = cv2.resize(
            image,
            self.image_size,
            interpolation=cv2.INTER_AREA
        )

        mask = cv2.resize(
            mask,
            self.image_size,
            interpolation=cv2.INTER_NEAREST
        )

        # Binary mask: background=0, smartphone=1
        mask = (mask > 127).astype(np.float32)

        # Data augmentation (training only)
        if self.augment:

            # Horizontal flip
            if np.random.random() < 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()

            # Vertical flip
            if np.random.random() < 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()

        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Convert HWC to CHW
        image = np.transpose(image, (2, 0, 1))

        # Convert to PyTorch tensors
        image = torch.from_numpy(
            np.ascontiguousarray(image)
        ).float()

        mask = torch.from_numpy(
            np.ascontiguousarray(mask)
        ).unsqueeze(0).float()

        return image, mask
