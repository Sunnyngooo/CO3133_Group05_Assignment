import torch
import json
from pathlib import Path
from assignment1.scripts.eda import main as eda_main

def main(model_name):   
    eda_main(model_name)
    ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
    output_dir = ASSIGNMENT_DIR / 'outputs' / 'eda' 
    
    # Check if files is created 
    assert (output_dir / "class_distribution.png").is_file()
    assert (output_dir / "sample_images.png").is_file()
    assert (output_dir / "dataset_stats.json").is_file()
    
    # Check content of files
    with open(output_dir / "dataset_stats.json") as f:
        stats = json.load(f)

    assert stats["num_samples"] == 51000
    assert stats["raw_image_shape"] == [28, 28]
    assert stats["model_input_image_shape"] == [1, 28, 28]
    assert sum(stats["class_counts"].values()) == 51000
    print("All tests passed !")
    
if __name__ == "__main__":
    main('cnn')  

    