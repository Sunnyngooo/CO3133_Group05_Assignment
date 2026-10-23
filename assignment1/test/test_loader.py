from pathlib import Path
from assignment1.src.utils.config import load_config
from assignment1.src.data.dataset import build_dataloaders

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
config = load_config('cnn')
train_loader, val_loader, test_loader = build_dataloaders(config)
img, label = next(iter(train_loader))
print(img.shape)
print(label.shape)
print(img.dtype)

assert len(train_loader.dataset) == 51000
assert len(val_loader.dataset) == 9000
assert len(test_loader.dataset) == 10000
assert img.ndim == 4
