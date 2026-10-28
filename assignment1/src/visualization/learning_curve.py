import matplotlib.pyplot as plt
from pathlib import Path
def plot_learning_curve(history_dict, model_name, save_path):
    """
    Plot loss and Macro-F1 of training and validation along epoches.
    Input: 
    - history: dictionary from history.json
    - model_name
    - save_path: path to the file saving the plots. 
    """
    history = history_dict['history']
    if not history:
        raise ValueError("Cannot plot learning curves since history is empty.")
    
    epoches = [record['epoch'] for record in history]
    train_loss =  [record['train_metrics']['loss'] for record in history]
    val_loss =  [record['val_metrics']['loss'] for record in history]
    train_f1 =  [record['train_metrics']['macro_f1'] for record in history]
    val_f1 =  [record['val_metrics']['macro_f1'] for record in history]
    
    fig, axes = plt.subplots(1,2, figsize=(12,6))
    
    axes[0].plot(epoches, train_loss, label="Training")
    axes[0].plot(epoches, val_loss, label="Validation")
    axes[0].set(title=f"{model_name} loss", xlabel="Epoch", ylabel='Loss')
    axes[0].grid(True, alpha=0.4)
    axes[0].legend()
    
    axes[1].plot(epoches, train_f1, label="Training")
    axes[1].plot(epoches, val_f1, label="Validation")
    axes[1].set(title=f"{model_name} macro-f1", xlabel="Epoch", ylabel='Macro-F1')
    axes[1].grid(True, alpha=0.4)
    axes[1].legend()
    
    best_epoch = history_dict['best_epoch']
    if best_epoch is not None:
        for ax in axes: 
            ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.7, label=f"Best epoch: {best_epoch}")
            ax.legend()
            
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig) 
