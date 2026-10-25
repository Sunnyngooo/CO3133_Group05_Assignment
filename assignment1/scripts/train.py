import torch
from pathlib import Path
from assignment1.src.utils.config import load_config, validate_config, save_config
from assignment1.src.utils.seed import set_seed
from assignment1.src.data.dataset import build_dataloaders
import argparse
import json
from assignment1.src.engine.trainer import fit
from assignment1.src.utils.run_id import get_run_id
from assignment1.src.models.factory import build_model

def main(model_name='cnn'):
    assignment_dir = Path(__file__).resolve().parents[1]
    # 1. Configuration
    config = load_config(model_name)
    validate_config(config)
    
    # 2. Reproducibility
    set_seed(seed=config['experiment']['seed'], deterministic=config['experiment']['deterministic'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # 3. Data pipeline 
    train_loader, val_loader, test_loader, stats = build_dataloaders(config)
    print("Train:", len(train_loader.dataset))
    print("Validation:", len(val_loader.dataset))
    print("Test:", len(test_loader.dataset))

    sample_img, sample_label = val_loader.dataset[0]

    print("Images:", sample_img.shape, sample_img.dtype)
    print("Labels:", sample_label, type(sample_label))
    print("Pixel range:", sample_img.min().item(), sample_img.max().item()) 
    
    # 4. Model initialization + loss + optimizer
    model = build_model(config=config).to(device)
    optimizer_name = config['training']['optimizer'].lower()
    learning_rate = config['training']['lr']
    weight_decay = config['training']['weight_decay']
    
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"{optimizer_name} is not supported.")
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # 5. Save the resolved config 
    config['data']['normalization'] = stats
    runs_root = assignment_dir / "runs"
    run_id = get_run_id(model_name, runs_root=runs_root)
    run_dir = runs_root / run_id
    
    config_path = run_dir / "config.yaml"
    history_path = run_dir / "history.json"
    checkpoint_path = assignment_dir / "checkpoints" / run_id / "best.pt"
    save_config(config=config, output_path= config_path)
    
    # 6. Fit 
    result = fit(model=model, train_loader=train_loader, val_loader=val_loader,
                 loss_fn=loss_fn, optimizer=optimizer, device=device,
                 epochs=config['training']['epochs'],
                 checkpoint_path=checkpoint_path, config=config)
    
    # 7. Save training result
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='cnn', choices=['linear', 'mlp', 'cnn', 'lstm', 'transformer'])
    args = parser.parse_args()
    
    main(model_name=args.model)