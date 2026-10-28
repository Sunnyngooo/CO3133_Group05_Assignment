import torch

from assignment2.models.deeplabv3 import DeepLabV3


def test_deeplabv3_forward():
    model = DeepLabV3(
        num_classes=19,
        pretrained_backbone=False,
    )

    model.eval()

    x = torch.randn(1, 3, 128, 256)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 19, 128, 256)


def test_deeplabv3_backward():
    model = DeepLabV3(
        num_classes=19,
        pretrained_backbone=False,
    )

    model.eval()

    x = torch.randn(1, 3, 64, 128)
    target = torch.randint(0, 19, (1, 64, 128))

    output = model(x)
    loss = torch.nn.functional.cross_entropy(output, target)

    loss.backward()

    assert model.model.classifier[-1].weight.grad is not None