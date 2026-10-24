import copy
import yaml
from pathlib import Path

def deep_merge(base: dict, override: dict) -> dict: 
    """Merge two dictionary"""
    origin = copy.deepcopy(base)
    for key, val in override.items():
        if key in origin and isinstance(val, dict) and isinstance(origin[key], dict):
            origin[key] = deep_merge(origin[key], val)
        else:
            origin[key] = copy.deepcopy(val)
            
    return origin

def load_config(model_name: str) -> dict: 
    """Load config dir and merge base.yaml with a model {model_name} config"""
    CONFIG_DIR = Path(__file__).resolve().parents[2] / 'configs'
    
    base_path = CONFIG_DIR / 'base.yaml'
    if not base_path.is_file():
        raise FileNotFoundError(f"Missing required base configuration file at {base_path}.")
    with open(base_path, encoding='utf-8') as f:
        try:
            base_config = yaml.safe_load(f) or {}
        except yaml.YAMLError as err:
            raise ValueError(f"Error in {base_path}: {err}.")
            
    model_path = CONFIG_DIR / f'{model_name}.yaml'
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing required model configuration file at {model_path}.")
    with open(model_path, encoding='utf-8') as f: 
        try:
            model_config = yaml.safe_load(f) or {}
        except yaml.YAMLError as err:
            raise ValueError(f"Error in {model_path}: {err}.")
        
    result_config = deep_merge(base_config, model_config)
    return result_config

def require_key(config: dict, section: str, key: str, expected_type):
    if section not in config: 
        raise ValueError(f"Missing required section: {section}.")
    
    if not isinstance(config[section], dict):
        raise TypeError(f"Section {section} must be a mapping.")
    
    if key not in config[section]: 
        raise ValueError(f"Missing required key: {section}.{key}.")
    
    value = config[section][key]
    if not isinstance(value, expected_type):
        raise TypeError(f"{section}.{key} must be {expected_type}, got {type(value).__name__}.")
    return value
    
def validate_config(config: dict) -> None: 
    """Check if the configuration is invalid"""        
    # environment section
    seed = require_key(config=config, section='experiment', key='seed', expected_type=int)
    if seed < 0:
        raise ValueError("experiment.seed must be non-negative.")
    
    require_key(config=config, section='experiment', key='deterministic', expected_type=bool)

    # data section
    val_ratio = require_key(config=config, section='data', key='val_ratio', expected_type=(int, float))    
    if not 0 < val_ratio < 1:
        raise ValueError("data.val_ratio must be in range (0,1).")
    
    batch_size = require_key(config=config, section='data', key='batch_size', expected_type=int) 
    if batch_size <= 0:
        raise ValueError("data.batch_size must be positive.")
    
    # training section
    training_epochs = require_key(config=config, section='training', key='epochs', expected_type=int)    
    if training_epochs <= 0:
        raise ValueError("training.epochs must be positive.")
      
    training_lr = require_key(config=config, section='training', key='lr', expected_type=(int, float))    
    if training_lr <= 0:
        raise ValueError("training.lr must be positive.") 
    
    allowed_optimizer = {"sgd", "adam", "adamw"}
    training_optimizer = require_key(config=config, section='training', key='optimizer', expected_type=str)
    if training_optimizer.lower() not in allowed_optimizer:
        raise ValueError("training.optimizer is not allowed")
    training_weight_decay = require_key(config=config, section='training', key='weight_decay', expected_type=(int, float))
    if training_weight_decay < 0: 
        raise ValueError("training.weight_decay must be positive.")
    
    # checkpoints section
    monitor = require_key(config=config, section='checkpoints', key='monitor', expected_type=str)
    mode = require_key(config=config, section='checkpoints', key='mode', expected_type=str) 
    if monitor != 'macro-f1' or mode != 'max': 
        raise ValueError("fit() only supports macro-f1 with mode=max.")
    
    # model section 
    models_avail = {'linear', 'mlp', 'cnn', 'lstm', 'transformer'}
    model_name = require_key(config=config, section='model', key='name', expected_type=str)
    if model_name not in models_avail:
        raise ValueError(f"Model {model_name} is not available. Retry with linear/mlp/cnn/lstm/transformer.")

def save_config(config: dict, output_path: Path): 
    if output_path.exists():
        raise FileExistsError(f"The config file {output_path} has already existed. Retry with a new directory (add timestamp or ID).")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding='utf-8') as f:
        yaml.safe_dump(config, f, 
                       default_flow_style=False,
                       sort_keys=False)