import torch
import time 
import numpy as np

def benchmark_inference(model, sample, device, warmup=20, repeats=200):
    """
    Measure inference time that model with learned weights predicts (only supported for CPU)
    Input: 
    - model: model loaded from best checkpoint
    - sample: one image sample (1,1,28,28)
    - device
    - warmup: number of run before of measurements
    - repeats: number of times measurement
    """
    if warmup < 0:
        raise ValueError("warmup must not be negative.")
    
    if repeats <= 0:
        raise ValueError("repeats must be larger than 0.")
    
    model = model.to(device)
    sample = sample.to(device)
    
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        
        times = []
        for _ in range(repeats):
            start_time = time.perf_counter()
            model(sample)
            elapsed_time = (time.perf_counter() - start_time) * 1000 # convert sec -> milliseconds
            times.append(elapsed_time)
        mean_ms = float(np.mean(times))
        median_ms = float(np.median(times))
        p95_ms = float(np.percentile(times, 95))
        min_ms = float(np.min(times))
        max_ms = float(np.max(times))
        
    return {
        "device": str(device),
        "batch_size": sample.shape[0],
        "warmup": warmup,
        "repeats": repeats,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "min_ms": min_ms,
        "max_ms": max_ms
    }