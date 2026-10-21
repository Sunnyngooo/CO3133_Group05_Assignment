
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from assignment2.dataset import CityscapesDataset
from assignment2.metrics import SegmentationMetrics
from assignment2.models.unet import UNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# def train_one_epoch(model, loader, optimizer, criterion, device):
def train_one_epoch(model, loader, optimizer, criterion, device, max_batches=None):
    model.train()
    total_loss = 0.0
    total_batches = 0

    # for images, masks in loader:
    for batch_idx, (images, masks) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
# def validate(model, loader, criterion, device, num_classes):
def validate(model, loader, criterion, device, num_classes, max_batches=None):
    model.eval()

    metrics = SegmentationMetrics(
        num_classes=num_classes,
        ignore_index=255,
    )

    total_loss = 0.0
    total_batches = 0

    # for images, masks in loader:
    for batch_idx, (images, masks) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        metrics.update(logits, masks)

        total_loss += loss.item()
        total_batches += 1

    results = metrics.compute()

    return (
        total_loss / max(total_batches, 1),
        results,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/assignment2.yaml",
    )
    parser.add_argument(
        "--data-root",
        default=None,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit batches for smoke testing only.",
    )

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    requested_device = cfg.get("device", "auto")
    if requested_device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(requested_device)

    print("Device:", device)

    data_root = args.data_root or cfg["dataset_path"]
    image_size = tuple(cfg["data"]["image_size"])
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]

    train_dataset = CityscapesDataset(
        data_root=data_root,
        split="train",
        image_size=image_size,
        augment=True,
    )

    val_dataset = CityscapesDataset(
        data_root=data_root,
        split="val",
        image_size=image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model_cfg = cfg["model"]

    model = UNet(
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
        base_channels=model_cfg["base_channels"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=cfg["training"]["ignore_index"]
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    checkpoint_dir = Path(cfg["checkpoint_dir"])
    output_dir = Path(cfg["output_dir"])

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = args.epochs or cfg["training"]["epochs"]
    best_miou = -1.0

    history_path = output_dir / "unet_history.csv"

    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "miou",
            "mean_dice",
        ])

        for epoch in range(1, epochs + 1):
            # train_loss = train_one_epoch(
            #     model, train_loader, optimizer, criterion, device
            # )

            # val_loss, results = validate(
            #     model,
            #     val_loader,
            #     criterion,
            #     device,
            #     model_cfg["num_classes"],
            # )
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion,
                device, args.max_batches
            )

            val_loss, results = validate(
                model, val_loader, criterion, device,
                model_cfg["num_classes"], args.max_batches
            )

            miou = results["miou"]
            mean_dice = results["mean_dice"]

            writer.writerow([
                epoch,
                train_loss,
                val_loss,
                miou,
                mean_dice,
            ])
            f.flush()

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"mIoU: {miou:.4f} | "
                f"Dice: {mean_dice:.4f}",
                flush=True,
            )

            if miou > best_miou:
                best_miou = miou

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_miou": best_miou,
                        "config": cfg,
                    },
                    checkpoint_dir / "unet_best.pth",
                )

    print("Training completed.")
    print("Best validation mIoU:", best_miou)


if __name__ == "__main__":
    main()
