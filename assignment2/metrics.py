
import torch


class SegmentationMetrics:
    def __init__(self, num_classes=19, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
        )

    @torch.no_grad()
    def update(self, logits, targets):
        predictions = logits.argmax(dim=1)

        predictions = predictions.reshape(-1).cpu()
        targets = targets.reshape(-1).cpu()

        valid = (
            (targets != self.ignore_index)
            & (targets >= 0)
            & (targets < self.num_classes)
        )

        predictions = predictions[valid]
        targets = targets[valid]

        indices = (
            targets * self.num_classes + predictions
        )

        counts = torch.bincount(
            indices,
            minlength=self.num_classes ** 2,
        )

        self.confusion_matrix += counts.reshape(
            self.num_classes,
            self.num_classes,
        )

    def compute(self):
        cm = self.confusion_matrix.float()

        tp = cm.diag()
        gt = cm.sum(dim=1)
        pred = cm.sum(dim=0)

        union = gt + pred - tp

        iou = torch.full(
            (self.num_classes,),
            float("nan"),
        )

        dice = torch.full(
            (self.num_classes,),
            float("nan"),
        )

        valid_iou = union > 0
        valid_dice = (gt + pred) > 0

        iou[valid_iou] = (
            tp[valid_iou] / union[valid_iou]
        )

        dice[valid_dice] = (
            2 * tp[valid_dice]
            / (gt[valid_dice] + pred[valid_dice])
        )

        return {
            "miou": torch.nanmean(iou).item(),
            "mean_dice": torch.nanmean(dice).item(),
            "per_class_iou": iou.tolist(),
            "per_class_dice": dice.tolist(),
            "confusion_matrix": self.confusion_matrix.clone(),
        }
