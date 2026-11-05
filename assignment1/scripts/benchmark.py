import torch
import json
import yaml
import argparse
from pathlib import Path
from assignment1.src.utils.config import validate_config
from assignment1.src.data.dataset import build_dataloaders
from assignment1.src.engine.checkpoint import load_checkpoint
from assignment1.src.models.factory import build_model
from assignment1.src.metrics.inference_benchmark import benchmark_inference
from assignment1.src.metrics.model_stats import count_parameters

def main(run_id, device='cpu', warmup=20, repeats=200):
    assignment_dir = Path(__file__).resolve().parents[1]
    run_dir = assignment_dir / "runs" / run_id
    config_path = run_dir / "config.yaml"
    checkpoint_path = assignment_dir / "checkpoints" / run_id / "best.pt"
    
    # 1. Check required files exist
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found at {config_path}.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}.")
    
    # 2. Load saved config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    validate_config(config)
    
    # 3. Build model
    model = build_model(config)
    
    # 4. Load checkpoint and restore model state 
    checkpoint = load_checkpoint(checkpoint_path, model=model, device=device, optimizer=None)
    
    # 5. Move model to device and evaluation mode
    model = model.to(device)
    model.eval()
    
    # 6. Take sample from test dataloader 
    _, _, test_loader, _ = build_dataloaders(config)
    
    x_batch, _ = next(iter(test_loader))
    sample = x_batch[: 1].to(device)        
    inference_benchmark = benchmark_inference(model=model, sample=sample, device=device, warmup=warmup, repeats=repeats)
    
    inference_metrics = {
        "run_id": run_id,
        "model_name": config['model']['name'],
        "device": inference_benchmark['device'],
        'parameter_count': count_parameters(model=model),
        'epoch': checkpoint['epoch'],
        'warmup': warmup, 
        'repeats': repeats,
        'mean_ms': inference_benchmark['mean_ms'],
        'median_ms': inference_benchmark['median_ms'],
        'p95_ms': inference_benchmark['p95_ms'], 
        'min_ms': inference_benchmark['min_ms'],
        'max_ms': inference_benchmark['max_ms'],
        'batch_size': inference_benchmark['batch_size'],
        'num_threads': torch.get_num_threads()          
    }
    
    # 7. Save metrics
    inf_metrics_path = run_dir / 'inference_metrics.json'
    with open(inf_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(inference_metrics, f, indent=4)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    
    main(run_id=args.run_id, device=args.device, warmup=args.warmup, repeats=args.repeats)