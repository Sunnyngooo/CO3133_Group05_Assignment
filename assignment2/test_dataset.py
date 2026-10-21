
import argparse

import torch
from torch.utils.data import DataLoader

from assignment2.dataset import (
    CityscapesDataset,
    NUM_CLASSES,
    IGNORE_INDEX,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    dataset = CityscapesDataset(
        data_root=args.data_root,
        split="train",
        image_size=(512, 1024),
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    images, masks = next(iter(loader))

    print("Dataset size:", len(dataset))
    print("Image shape:", images.shape)
    print("Mask shape:", masks.shape)
    print("Image dtype:", images.dtype)
    print("Mask dtype:", masks.dtype)
    print("Unique mask values:", torch.unique(masks).tolist())

    assert images.shape[1:] == (3, 512, 1024)
    assert masks.shape[1:] == (512, 1024)
    assert masks.dtype == torch.int64

    assert torch.all(
        (masks == IGNORE_INDEX)
        | ((masks >= 0) & (masks < NUM_CLASSES))
    )

    print("DataLoader test PASSED")


if __name__ == "__main__":
    main()
