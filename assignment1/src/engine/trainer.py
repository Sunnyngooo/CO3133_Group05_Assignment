import torch
import time
from pathlib import Path
from assignment1.src.metrics.cls_metrics import compute_classification_metrics
from assignment1.src.engine.evaluator import validate
from assignment1.src.engine.checkpoint import save_checkpoint

def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    
    running_loss = 0.0
    total = 0
    preds_epoch = []
    truth_epoch = []
    
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
          
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        # Total loss
        running_loss += loss.item() * x_batch.size(0)
        total += y_batch.size(0)
        
        _, y_pred = torch.max(logits, 1)
        
        y_batch = y_batch.detach().cpu().tolist()
        y_pred = y_pred.detach().cpu().tolist()
        truth_epoch.extend(y_batch)
        preds_epoch.extend(y_pred)
        
        if batch_idx % 200 == 0: 
            print(f"Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item(): .4f}")
    
    if total == 0:
        raise ValueError("Training loader cannot be empty.")
    epoch_loss = running_loss / total
    metrics = compute_classification_metrics(truth_epoch, preds_epoch)
    metrics['loss'] = epoch_loss
    return metrics

def fit(model, train_loader, val_loader, loss_fn, optimizer, device, epochs, checkpoint_path: Path, config):
    if isinstance(epochs, bool) or not isinstance(epochs, int): 
        raise TypeError("epochs must be integer.")    
    if epochs < 1: 
        raise ValueError("epochs must be larger than 1.")
    
    if not isinstance(checkpoint_path, Path):
        raise TypeError(f"Checkpoint path must be pathlib.Path object: {checkpoint_path}")
    # Init training state
    history = []
    best_macro_f1 = float('-inf')
    best_epoch = None
    start_time = time.perf_counter()
    
    # training loop
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model=model, train_loader=train_loader, 
                                  loss_fn=loss_fn,optimizer=optimizer, device=device)
        val_metrics = validate(model=model, val_loader=val_loader, loss_fn=loss_fn, device=device)
        
        # Save training history
        history.append({'epoch': epoch, 'train_metrics': train_metrics, 'val_metrics': val_metrics})
        
        # Choose the best checkpoint
        if val_metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = val_metrics['macro_f1']
            best_epoch = epoch
            save_checkpoint(checkpoint_path, model, optimizer, best_epoch, val_metrics, config)
        
        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"Accuracy: {train_metrics['accuracy']:.4f} "
            f"Macro-F1: {train_metrics['macro_f1']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} "
            f"Accuracy: {val_metrics['accuracy']:.4f} "
            f"Macro F1: {val_metrics['macro_f1']:.4f} | "
            f"Best Macro F1: {best_macro_f1:.4f} (Epoch {best_epoch})"
        )
    elapsed_time = time.perf_counter() - start_time 
    
    return {
        "history": history,
        'best_epoch':  best_epoch,
        'best_macro_f1': best_macro_f1,
        'best_checkpoint_path': str(checkpoint_path),
        'training_time': elapsed_time
    }