import torch
import json
import yaml
import argparse
import math
from pathlib import Path
from assignment1.src.utils.config import validate_config
from assignment1.src.data.dataset import build_dataloaders
from assignment1.src.engine.checkpoint import load_checkpoint
from assignment1.src.models.factory import build_model
from assignment1.src.engine.predictions import collect_predictions
from assignment1.src.metrics.error_analysis import analyze_classification_errors
from assignment1.src.visualization.confusion_plots import plot_confusion_matrix

def main(run_id, device='cpu'):
    assignment_dir = Path(__file__).resolve().parents[1]
    run_dir = assignment_dir / "runs" / run_id
    config_path = run_dir / "config.yaml"
    test_metrics_path = run_dir / 'test_metrics.json'
    checkpoint_path = assignment_dir / "checkpoints" / run_id / "best.pt"
    
    # 1. Check required files exist
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found at {config_path}.")
    if not test_metrics_path.is_file():
        raise FileNotFoundError(f"Test metrics not found at {test_metrics_path}.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}.")
    
    # 2. Load saved config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    validate_config(config)
    
    # 3. Build model
    model = build_model(config)
    model_name = config['model']['name']
    
    # 4. Load checkpoint and restore model state 
    checkpoint = load_checkpoint(checkpoint_path, model=model, device=device, optimizer=None)
    
    # 5. Move model to device and evaluation mode
    model = model.to(device)
    model.eval()
    
    # 6. Take sample from test dataloader 
    _, _, test_loader, _ = build_dataloaders(config)
    class_names = test_loader.dataset.classes
    # 7. Collect predictions
    predictions = collect_predictions(model=model, dataloader=test_loader, device=device)
    y_true = predictions['y_true']
    y_pred = predictions['y_pred']
    
    # 8. Calculate error analysis and verifies accuracy with test metrics'
    error_analysis = analyze_classification_errors(y_true=y_true, y_pred=y_pred, class_names=class_names)
    
    with open(test_metrics_path, 'r', encoding='utf-8') as f:
        test_metrics = json.load(f)
    
    if not math.isclose(error_analysis['accuracy'], test_metrics['test_metrics']['accuracy'], rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"Accuracy mismatch: error analysis: {error_analysis['accuracy']}; test metrics: {test_metrics['test_metrics']['accuracy']}")
    if not math.isclose(error_analysis['macro_f1'], test_metrics['test_metrics']['macro_f1'], rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"Macro-F1 mismatch: error analysis: {error_analysis['macro_f1']}; test metrics: {test_metrics['test_metrics']['macro_f1']}")
    
    # 9. Plot confusion matrices
    save_path = assignment_dir / 'reports' / 'figures' 
    save_path.mkdir(parents=True, exist_ok=True)
    
    plot_confusion_matrix(cm=error_analysis['confusion_matrix'], class_names=class_names, save_path= save_path / f"{model_name}_cm_raw.png")
    plot_confusion_matrix(cm=error_analysis['confusion_matrix'], class_names=class_names, save_path= save_path / f"{model_name}_cm_normalized.png", normalize=True)
    # 10. Save error analysis 
    result = {
        'run_id': run_id,
        'model_name': model_name,
        'checkpoint_epoch': checkpoint['epoch'],
        'accuracy': error_analysis['accuracy'],
        'macro_f1': error_analysis['macro_f1'],
        'confusion_matrix': error_analysis['confusion_matrix'],
        'per_class_metrics': error_analysis['per_class_metrics'],
        'top_confusions': error_analysis['top_confusions']
    }
    
    with open(run_dir / 'error_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--device", type=str, default='cpu')
    args = parser.parse_args()
    
    main(run_id=args.run_id, device=args.device)