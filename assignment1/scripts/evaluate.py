import torch
import json
import yaml
import re
import argparse
from pathlib import Path
from assignment1.src.utils.config import validate_config
from assignment1.src.utils.seed import set_seed
from assignment1.src.data.dataset import build_dataloaders
from assignment1.src.engine.checkpoint import load_checkpoint
from assignment1.src.engine.evaluator import test as test_model

from assignment1.test.dummy_model import DummyModel

def main(run_id):
    if not re.fullmatch(f"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid run_id.")
    # 1. Get paths
    assignment_dir = Path(__file__).resolve().parents[1]
    run_dir = assignment_dir / "runs" / run_id
    config_path = run_dir / "config.yaml"
    checkpoint_path = assignment_dir / "checkpoints" / run_id / "best.pt"
    # Check required files exist
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found at {config_path}.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}.")
    
    # 2. Load saved config
    with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    validate_config(config)
    
    # 3. Reproducibility and device
    set_seed(seed=config['experiment']['seed'], deterministic=config['experiment']['deterministic'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 4. Build test loader
    _, _, test_loader, _ = build_dataloaders(config=config)
    
    # 5. Model
    model = DummyModel().to(device)
    
    # 6. Load best checkpoint
    checkpoint = load_checkpoint(path=checkpoint_path, model=model,
                                 device=device, optimizer=None)
    
    if checkpoint["config"] != config:    
        raise ValueError("Checkpoint config does not match saved run config.")
    # 7. Evaluate test set
    loss_fn = torch.nn.CrossEntropyLoss()
    test_metrics = test_model(model=model, test_loader=test_loader, 
                            loss_fn=loss_fn, device=device)
    # 8. Save test results 
    result = {
        "run_id": run_id,
        "checkpoint_epoch": checkpoint['epoch'],
        'checkpoint_val_macro_f1': checkpoint['val_metrics']['macro_f1'],
        'test_metrics': test_metrics
    }
    
    test_metrics_path = run_dir / 'test_metrics.json'
    with open(test_metrics_path, 'w', encoding='utf-8') as f: 
        json.dump(result, f, indent=4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    args = parser.parse_args()
    
    main(run_id=args.run_id)