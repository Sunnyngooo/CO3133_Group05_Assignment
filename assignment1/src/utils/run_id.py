from pathlib import Path 

def get_run_id(model_name: str, runs_root: Path) -> str:
    index = 1
    while True:
        run_id = f"{model_name}_baseline_{index:03d}"
        
        if not (runs_root / run_id).exists():
            return run_id
        index += 1