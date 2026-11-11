import csv
import json
import yaml
import argparse
from pathlib import Path
from assignment1.src.utils.config import validate_config
from assignment1.src.visualization.learning_curve import plot_learning_curve

def collect_run_summary(run_id, runs_root):
    """
    Collect files for each run_id (config.yaml, history.json, test_metrics.json, inference_metrics.json)
    in order to collect metadata.
    """
    run_dir = runs_root / run_id
    config_path = run_dir / 'config.yaml'
    history_path = run_dir / 'history.json'
    inference_path = run_dir / 'inference_metrics.json'
    test_path = run_dir / 'test_metrics.json'
    
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found at {config_path}.")
    if not history_path.is_file():
        raise FileNotFoundError(f"History not found at {history_path}.")
    if not inference_path.is_file():
        raise FileNotFoundError(f"Inference metrics not found at {inference_path}.")
    if not test_path.is_file():
        raise FileNotFoundError(f"Test metrics not found at {test_path}.")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    validate_config(config)
    
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
        
    with open(inference_path, 'r', encoding='utf-8') as f:
        inference_metrics = json.load(f)
    
    with open(test_path, 'r', encoding='utf-8') as f:
        test_metrics  = json.load(f)
        
    # Check consistency 
    if config['model']['name'] != inference_metrics['model_name']:
        raise ValueError(f"The model in config is different from the one in inference metrics: Config - {config['model']['name']}; Inference metrics: {inference_metrics['model_name']}.")
    
    if not(test_metrics['run_id'] == inference_metrics['run_id'] == run_id):
        raise ValueError(f"The run_id do not match: run_id - {run_id}; Test metrics - {test_metrics['run_id']}; Inference metrics: {inference_metrics['run_id']}.")
    
    if not (history['best_epoch'] == test_metrics['checkpoint_epoch'] == inference_metrics['epoch']):
        raise ValueError(f"The best epoch do not match: history - {history['best_epoch']}; test_metrics - {test_metrics['checkpoint_epoch']}; inference_metrics - {inference_metrics['epoch']}.")
    
    return {
        "model_name": config['model']['name'], 
        "run_id": run_id,
        'test_accuracy': test_metrics['test_metrics']['accuracy'],
        'test_macro_f1': test_metrics['test_metrics']['macro_f1'], 
        'test_loss': test_metrics['test_metrics']['loss'], 
        "best_epoch": history['best_epoch'], 
        'best_val_macro_f1': history['best_macro_f1'], 
        'total_params': inference_metrics['parameter_count']['total_params'], 
        'training_time_s': history['training_time'], 
        'median_inference_ms': inference_metrics['median_ms'], 
        'p95_inference_ms': inference_metrics['p95_ms']
    }
def build_comparison(run_ids):
    assignment_dir = Path(__file__).resolve().parents[1]
    runs_root = assignment_dir / 'runs'
    runs_summary = []
    for run_id in run_ids:
        runs_summary.append(collect_run_summary(run_id=run_id, runs_root=runs_root))
    runs_summary.sort(key=lambda summary: summary['test_macro_f1'], reverse=True)
    
    csv_data = [{
        "Model": summary['model_name'],
        "Test Accuracy": summary['test_accuracy'],
        "Test Macro-F1": summary['test_macro_f1'],
        "Parameters": summary['total_params'],
        'Training Time (s)': summary['training_time_s'], 
        'Inference Time (Median) (ms)': summary['median_inference_ms']
    } 
                for summary in runs_summary]
    report_dir = assignment_dir / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / 'comparison.csv', 'w', newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Model', 'Test Accuracy', 'Test Macro-F1', 'Parameters', 'Training Time (s)', 'Inference Time (Median) (ms)'])
        writer.writeheader()
        writer.writerows(csv_data)
        
    figure_dir = report_dir / 'figures'
    
    for summary in runs_summary:
        run_id = summary['run_id']
        
        history_path = runs_root / run_id / 'history.json'
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        model_name = summary['model_name']
        
        plot_learning_curve(history_dict=history, model_name=model_name, save_path=figure_dir / f'{run_id}_learning_curves.png')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-ids", type=str, nargs="+", required=True)
    args = parser.parse_args()
    
    build_comparison(run_ids=args.run_ids)