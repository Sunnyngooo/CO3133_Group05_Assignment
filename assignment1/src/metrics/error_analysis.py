import numpy as np
from assignment1.src.metrics.cls_metrics import compute_confusion_matrix, compute_per_class_metrics, compute_classification_metrics

def get_top_confusions(cm, class_names, top_k=5):
    if cm.shape != (len(class_names), len(class_names)):
        raise ValueError("The confusion matrix does not match the length of class names.")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k must be integer.")
    if top_k <=0:
        raise ValueError("top_k must be positive.")
    
    confusions =[]
    for row_idx, row in enumerate(cm):
        for idx, count in enumerate(row):
            if row_idx != idx and count > 0: 
                confusions.append({
                    'true_class': class_names[row_idx],
                    'pred_class': class_names[idx],
                    'count': int(count) # since count is np.int64
                })
    confusions.sort(key=lambda confusion: confusion['count'], reverse=True)
    return confusions[: top_k]
            
def analyze_classification_errors(y_true, y_pred, class_names):
    cm = compute_confusion_matrix(y_true=y_true, y_pred=y_pred, num_classes=len(class_names))
    
    metrics_per_class = compute_per_class_metrics(y_true=y_true, y_pred=y_pred, num_classes=len(class_names))
    summary_per_class = []
    for idx, class_name in enumerate(class_names):
        summary = {
            'class_id': idx,
            'class_name': class_name,
            'precision': metrics_per_class['precision'][idx],
            'recall': metrics_per_class['recall'][idx],
            'f1': metrics_per_class['f1'][idx],
            'support': metrics_per_class['support'][idx],
        }
        summary_per_class.append(summary)
    top_confusions = get_top_confusions(cm, class_names, 5)
    
    classification_metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred, num_classes=len(class_names))

    return {
        'accuracy': float(classification_metrics['accuracy']),
        'macro_f1': float(classification_metrics['macro_f1']),
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': summary_per_class,
        'top_confusions': top_confusions
    }