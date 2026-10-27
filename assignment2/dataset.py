
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


NUM_CLASSES = 19
IGNORE_INDEX = 255

# Official Cityscapes labelId -> trainId mapping
LABEL_TO_TRAIN = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4,
    17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13,
    27: 14, 28: 15, 31: 16, 32: 17,
    33: 18,
}

LABEL_LUT = np.full(256, IGNORE_INDEX, dtype=np.uint8)

for label_id, train_id in LABEL_TO_TRAIN.items():
    LABEL_LUT[label_id] = train_id


class CityscapesDataset(Dataset):
    def __init__(
        self,
        data_root,
        split="train",
        image_size=(512, 1024),
        augment=False,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.augment = augment

        image_dir = (
            self.data_root
            / "leftImg8bit_trainvaltest"
            / "leftImg8bit"
            / split
        )

        mask_dir = (
            self.data_root
            / "gtFine_trainvaltest"
            / "gtFine"
            / split
        )

        self.samples = []

        for image_path in sorted(
            image_dir.rglob("*_leftImg8bit.png")
        ):
            city = image_path.parent.name

            stem = image_path.name.replace(
                "_leftImg8bit.png", ""
            )

            mask_path = (
                mask_dir
                / city
                / f"{stem}_gtFine_labelIds.png"
            )

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Missing mask: {mask_path}"
                )

            self.samples.append((image_path, mask_path))

        if not self.samples:
            raise RuntimeError(
                f"No Cityscapes images found in {image_dir}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]

        with Image.open(image_path) as im:
            image = im.convert("RGB")

        with Image.open(mask_path) as m:
            mask = m.copy()

        # Resize while preserving Cityscapes 2:1 aspect ratio
        image = TF.resize(
            image,
            self.image_size,
            interpolation=InterpolationMode.BILINEAR,
        )

        mask = TF.resize(
            mask,
            self.image_size,
            interpolation=InterpolationMode.NEAREST,
        )

        # Apply identical augmentation to image and mask
        if self.augment and random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        mask_np = np.asarray(mask, dtype=np.uint8)
        mask_np = LABEL_LUT[mask_np].astype(np.int64)

        image = TF.to_tensor(image)
        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        mask = torch.from_numpy(mask_np.copy())

        return image, mask
