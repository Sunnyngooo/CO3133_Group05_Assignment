from pathlib import Path
from assignment1.src.utils.config import load_config, validate_config, save_config
from assignment1.src.utils.seed import set_seed
from assignment1.src.data.dataset import build_dataloaders

def main(model_name='cnn'):
    ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
    # 1. Configuration
    config = load_config(model_name)
    validate_config(config)
    
    # 2. Reproducibility
    set_seed(seed=config['experiment']['seed'], deterministic=config['experiment']['deterministic'])

    # 3. Data pipeline 
    train_loader, val_loader, test_loader, stats = build_dataloaders(config)
    print("Train:", len(train_loader.dataset))
    print("Validation:", len(val_loader.dataset))
    print("Test:", len(test_loader.dataset))

    images, labels = next(iter(train_loader))

    print("Images:", images.shape, images.dtype)
    print("Labels:", labels.shape, labels.dtype)
    print("Pixel range:", images.min().item(), images.max().item())
    
    # 4. Save the resolved config 
    config['data']['normalization'] = stats
    run_dir =  ASSIGNMENT_DIR / "runs" / f"{model_name}_baseline_001" / "config.yaml"
    save_config(config=config, output_path= run_dir)
    
    # 4. Model initialization
    
    # 5. Training 
    
    # 6. Final evaluation 
    
if __name__ == "__main__":
    main()