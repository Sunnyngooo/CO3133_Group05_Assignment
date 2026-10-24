import torch
from assignment1.src.metrics.cls_metrics import compute_classification_metrics, validate_label_vector

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