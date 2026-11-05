import matplotlib.pyplot as plt 
from pathlib import Path 

def select_prediction_samples(y_true, y_pred, n_correct, n_incorrect): 
    corrects = []
    incorrects = []
    
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            corrects.append(i)
        else:
            incorrects.append(i)
    return {
        "correct": corrects[: n_correct],
        "incorrect": incorrects[: n_incorrect]
    }
    
def plot_prediction_samples(dataset, y_true, y_pred, selected_indices, class_names, mean, std, save_path):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), layout='constrained')
    groups = [
        selected_indices['correct'],
        selected_indices['incorrect'],
    ]
    
    for row, indices in enumerate(groups):
        for col, ax in enumerate(axes[row]):
            if col >= len(indices):
                ax.axis('off')
                continue
            
            index = indices[col]
            image, _ = dataset[index]
            image = image.detach().cpu().squeeze(0) * std + mean
            
            ax.imshow(image.numpy(), cmap='gray')
            ax.set_title(
                f"True: {class_names[int(y_true[index])]}\n"
                f"Pred: {class_names[int(y_pred[index])]}",
                fontsize=9,
                pad=8
            )
            ax.axis('off')
            
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    