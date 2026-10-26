import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_confusion_matrix(cm, class_names, save_path, normalize=False):
    cm = np.asarray(cm)
    if cm.shape != (len(class_names), len(class_names)):
        raise ValueError("The confusion matrix does not match the length of class names.")
    
    if normalize:
        rows_sum = cm.sum(axis=1, keepdims=True)
        plot_cm = np.divide(cm, rows_sum, out=np.zeros_like(cm, dtype=float), where=rows_sum != 0)
        annotation_format = '.2f'
    else: 
        plot_cm = cm
        annotation_format = 'd'
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(plot_cm, annot=True, fmt=annotation_format, cmap='Blues', cbar=True, xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)