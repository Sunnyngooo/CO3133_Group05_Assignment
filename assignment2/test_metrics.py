
import torch

from assignment2.metrics import SegmentationMetrics


def main():
    metrics = SegmentationMetrics(
        num_classes=3,
        ignore_index=255,
    )

    targets = torch.tensor([
        [
            [0, 1],
            [2, 255],
        ]
    ])

    predictions = torch.tensor([
        [
            [0, 1],
            [1, 2],
        ]
    ])

    logits = torch.nn.functional.one_hot(
        predictions,
        num_classes=3,
    ).permute(0, 3, 1, 2).float()

    metrics.update(logits, targets)
    results = metrics.compute()

    print("mIoU:", results["miou"])
    print("Mean Dice:", results["mean_dice"])
    print("Per-class IoU:", results["per_class_iou"])
    print("Confusion matrix:")
    print(results["confusion_matrix"])

    expected_iou = (1.0 + 0.5 + 0.0) / 3
    expected_dice = (1.0 + 2 / 3 + 0.0) / 3

    assert abs(results["miou"] - expected_iou) < 1e-6
    assert abs(results["mean_dice"] - expected_dice) < 1e-6

    print("Metrics test PASSED")


if __name__ == "__main__":
    main()
