import torch
from assignment1.src.metrics.cls_metrics import compute_classification_metrics


    
def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    
    running_loss = 0.0
    total = 0
    
    truth_epoch = []
    preds_epoch = []
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            
            # Total loss
            running_loss += loss.item() * x_batch.size(0)            
            total += y_batch.size(0)
            
            _, y_pred = torch.max(logits, 1)
            
            y_batch = y_batch.detach().cpu().tolist()
            y_pred = y_pred.detach().cpu().tolist()
            truth_epoch.extend(y_batch)
            preds_epoch.extend(y_pred)
    
    if total == 0:
        raise ValueError("Validation loader is empty.")
    epoch_loss = running_loss / total
    metrics = compute_classification_metrics(truth_epoch, preds_epoch)
    metrics['loss'] = epoch_loss
    return metrics

def validate(model, val_loader, loss_fn, device):
    return evaluate(model=model, data_loader=val_loader, device=device, loss_fn=loss_fn)
    
def test(model, test_loader, loss_fn, device):
    return evaluate(model=model, data_loader=test_loader, device=device, loss_fn=loss_fn)