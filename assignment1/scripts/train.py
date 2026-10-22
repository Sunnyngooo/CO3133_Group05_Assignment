from pathlib import Path
from assignment1.src.utils.config import load_config, validate_config, save_config
from assignment1.src.utils.seed import set_seed
from assignment1.src.data.dataset import build_dataloaders

def main():
    ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
    # 1. Configuration
    config = load_config('cnn')
    validate_config(config)
    save_config(config=config, output_path=  ASSIGNMENT_DIR / "runs" / "cnn_baseline_001" / "config.yaml")
    
    # 2. Reproducibility
    set_seed(seed=config['experiment']['seed'], deterministic=config['experiment']['deterministic'])

    # 3. Data pipeline 
    train_loader, val_loader, test_loader = build_dataloaders(config)
    print("Train:", len(train_loader.dataset))
    print("Validation:", len(val_loader.dataset))
    print("Test:", len(test_loader.dataset))

    images, labels = next(iter(train_loader))

    print("Images:", images.shape, images.dtype)
    print("Labels:", labels.shape, labels.dtype)
    print("Pixel range:", images.min().item(), images.max().item())
    # 4. Model initialization
    
    # 5. Training 
    
    # 6. Final evaluation 
    
if __name__ == "__main__":
    main()