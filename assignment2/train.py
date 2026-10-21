import argparse
import csv
import json
import random
import time
import shutil
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from assignment2.dataset import CityscapesDataset
from assignment2.metrics import SegmentationMetrics
from assignment2.models.unet import UNet


CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation",
    "terrain", "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(requested):
    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    return device


def make_autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    return nullcontext()


def train_one_epoch(
    model, loader, optimizer, criterion,
    scaler, device, use_amp, max_batches=None
):
    model.train()

    total_loss = 0.0
    total_valid_pixels = 0
    start_time = time.perf_counter()

    for batch_idx, (images, masks) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with make_autocast(device, use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        valid_pixels = (masks != criterion.ignore_index).sum().item()

        total_loss += loss.item() * valid_pixels
        total_valid_pixels += valid_pixels

    if total_valid_pixels == 0:
        raise RuntimeError("No valid training pixels")

    elapsed = time.perf_counter() - start_time

    return total_loss / total_valid_pixels, elapsed


@torch.no_grad()
def validate(
    model, loader, criterion, device,
    num_classes, ignore_index,
    use_amp, max_batches=None
):
    model.eval()

    metrics = SegmentationMetrics(
        num_classes=num_classes,
        ignore_index=ignore_index,
    )

    total_loss = 0.0
    total_valid_pixels = 0

    for batch_idx, (images, masks) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with make_autocast(device, use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        metrics.update(logits, masks)

        valid_pixels = (masks != ignore_index).sum().item()

        total_loss += loss.item() * valid_pixels
        total_valid_pixels += valid_pixels

    if total_valid_pixels == 0:
        raise RuntimeError("No valid validation pixels")

    return total_loss / total_valid_pixels, metrics.compute()


def save_checkpoint(
    path, model, optimizer, scaler,
    epoch, best_miou, config, generator
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_miou": best_miou,
        "config": config,
        "data_generator_state": generator.get_state(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None
            ),
        },
    }

    # Save to temporary file first.
    tmp_path = path.with_suffix(".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(
    path, model, optimizer, scaler, device, generator
):    # Only load checkpoints created by this project and trusted sources.
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    rng = checkpoint.get("rng_state")

    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])

        if torch.cuda.is_available() and rng["cuda"] is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])

    if "data_generator_state" in checkpoint:
        generator.set_state(checkpoint["data_generator_state"])

    return checkpoint["epoch"], checkpoint["best_miou"]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", default="configs/assignment2.yaml"
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)

    args = parser.parse_args()

    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be >= 1")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    set_seed(seed)

    device = get_device(cfg.get("device", "auto"))
    use_amp = device.type == "cuda"

    print("Device:", device)
    print("AMP enabled:", use_amp)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.reset_peak_memory_stats()

    data_root = args.data_root or cfg["dataset_path"]

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    image_size = tuple(data_cfg["image_size"])
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else data_cfg["batch_size"]
    )

    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else data_cfg["num_workers"]
    )

    if batch_size < 1 or num_workers < 0:
        parser.error("Invalid batch size or worker count")

    generator = torch.Generator()
    generator.manual_seed(seed)

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
        worker_init_fn=seed_worker,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
    )

    model = UNet(
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
        base_channels=model_cfg["base_channels"],
    ).to(device)

    ignore_index = train_cfg["ignore_index"]

    criterion = nn.CrossEntropyLoss(
        ignore_index=ignore_index,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp
    )

    smoke_test = args.max_batches is not None

    if args.output_root:
        root = Path(args.output_root)
        checkpoint_dir = root / "checkpoints"
        output_dir = root / "outputs"
    else:
        checkpoint_dir = Path(cfg["checkpoint_dir"])
        output_dir = Path(cfg["output_dir"])

    if smoke_test:
        checkpoint_dir = checkpoint_dir / "smoke_test"
        output_dir = output_dir / "smoke_test"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config_path = output_dir / "run_config.yaml"

    if args.resume:
        if not run_config_path.exists():
            raise FileNotFoundError(
                f"Run config not found: {run_config_path}"
            )

        with open(run_config_path, "r", encoding="utf-8") as f:
            previous_config = yaml.safe_load(f)

        if previous_config != cfg:
            raise RuntimeError(
                "Current config differs from the original run config. "
                "Use the original configuration when resuming."
            )
    else:
        with open(run_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                cfg,
                f,
                sort_keys=False,
                allow_unicode=True,
            )

    epochs = (
        args.epochs
        if args.epochs is not None
        else train_cfg["epochs"]
    )

    start_epoch = 1
    best_miou = -1.0

    if args.resume:
        last_epoch, best_miou = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scaler,
            device,
            generator,
        )
        start_epoch = last_epoch + 1
        print(f"Resuming from epoch {start_epoch}")

    history_path = output_dir / "unet_history.csv"

    if args.resume:
        if not history_path.exists():
            raise FileNotFoundError(
                f"History CSV not found: {history_path}"
            )

        with open(history_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            raise RuntimeError("History CSV is empty")

        last_history_epoch = int(rows[-1]["epoch"])
        checkpoint_epoch = start_epoch - 1

        if last_history_epoch != checkpoint_epoch:
            raise RuntimeError(
                "Checkpoint/history mismatch: "
                f"checkpoint epoch={checkpoint_epoch}, "
                f"history epoch={last_history_epoch}. "
                "Restore matching checkpoint and history files."
            )

    append_history = args.resume is not None

    with open(
        history_path,
        "a" if append_history else "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        if not append_history:
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "miou",
                "mean_dice",
                "epoch_seconds",
            ])

        for epoch in range(start_epoch, epochs + 1):
            train_loss, elapsed = train_one_epoch(
                model, train_loader, optimizer, criterion,
                scaler, device, use_amp, args.max_batches,
            )

            val_loss, results = validate(
                model, val_loader, criterion, device,
                model_cfg["num_classes"], ignore_index,
                use_amp, args.max_batches,
            )

            miou = results["miou"]
            mean_dice = results["mean_dice"]


            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | "
                f"mIoU: {miou:.4f} | "
                f"Dice: {mean_dice:.4f} | "
                f"Time: {elapsed:.1f}s",
                flush=True,
            )

            per_class = {
                name: {
                    "iou": results["per_class_iou"][i],
                    "dice": results["per_class_dice"][i],
                }
                for i, name in enumerate(CLASS_NAMES)
            }

            with open(
                output_dir / f"unet_per_class_epoch_{epoch}.json",
                "w",
                encoding="utf-8",
            ) as metrics_file:
                json.dump(per_class, metrics_file, indent=2)

            improved = miou > best_miou
            if improved:
                best_miou = miou

            save_checkpoint(
                checkpoint_dir / "unet_last.pth",
                model, optimizer, scaler,
                epoch, best_miou, cfg, generator,
            )

            if improved:
                save_checkpoint(
                    checkpoint_dir / "unet_best.pth",
                    model, optimizer, scaler,
                    epoch, best_miou, cfg, generator,
                )

            writer.writerow([
                epoch,
                train_loss,
                val_loss,
                miou,
                mean_dice,
                elapsed,
            ])
            f.flush()

            if device.type == "cuda":
                peak_gb = (
                    torch.cuda.max_memory_allocated()
                    / 1024**3
                )
                print(f"Peak GPU memory: {peak_gb:.2f} GB")

    print("Training completed.")
    print("Best validation mIoU:", best_miou)


if __name__ == "__main__":
    main()
