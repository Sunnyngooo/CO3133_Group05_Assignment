
import torch
import torch.nn as nn

from assignment2.models.unet import UNet


def main():
    torch.manual_seed(43)

    model = UNet(
        in_channels=3,
        num_classes=19,
        base_channels=32,
    )

    # Small input to keep the test inexpensive on CPU.
    images = torch.randn(1, 3, 128, 256)

    logits = model(images)

    print("Input shape:", images.shape)
    print("Output shape:", logits.shape)

    assert logits.shape == (1, 19, 128, 256)

    targets = torch.randint(
        low=0,
        high=19,
        size=(1, 128, 256),
    )

    # Verify ignore_index support.
    targets[:, :16, :] = 255

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    loss = criterion(logits, targets)
    loss.backward()

    print("Loss:", loss.item())

    assert torch.isfinite(loss)

    assert any(
        p.grad is not None
        for p in model.parameters()
        if p.requires_grad
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Parameters:", f"{parameters:,}")
    print("Forward/backward test PASSED")


if __name__ == "__main__":
    main()
