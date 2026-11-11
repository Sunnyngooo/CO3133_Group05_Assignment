import torch
def collect_predictions(model, dataloader, device):
    "Collect y_pred and y_true prepare for error analysis"
    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        y_trues = []
        y_preds = []
        
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            _, y_pred = torch.max(logits, dim=1)
            
            y_pred = y_pred.detach().cpu().tolist()
            y_batch = y_batch.detach().cpu().tolist()
            y_trues.extend(y_batch)
            y_preds.extend(y_pred)
            
    return {
        'y_true': y_trues,
        'y_pred': y_preds
    }