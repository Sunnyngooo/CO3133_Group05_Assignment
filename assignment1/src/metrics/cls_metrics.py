import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support

def validate_label_vector(y_values, name):
    if isinstance(y_values, torch.Tensor):
        y_values = y_values.detach().cpu().numpy()
    y_values = np.asarray(y_values)
     
    if y_values.ndim != 1: 
        raise ValueError(f"{name} must be a 1-dimension vector.")
    
    if y_values.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    
    if not np.issubdtype(y_values.dtype, np.integer): 
        raise ValueError(f"{name} must have integer labels.")
    return y_values
    
def validate_input(y_true, y_pred, num_classes):
    # validate num_classes 
    if isinstance(num_classes, (bool, np.bool_)) or not isinstance(num_classes, (int, np.integer)):
        raise TypeError("num_classes must be an integer.")
    
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    
    # Validate y
    y_true = validate_label_vector(y_true, "y_true")
    y_pred = validate_label_vector(y_pred, "y_pred")
    
    if y_true.size != y_pred.size: 
        raise ValueError("Size of y_true and y_pred do not match.")
    
    for labels, name in ((y_pred, "y_pred"), (y_true, "y_true")):
        if np.any(labels < 0) or np.any(labels >= num_classes):
            raise ValueError(f"{name} must have labels in range [0, {num_classes - 1}].")
    return y_true, y_pred

def compute_classification_metrics(y_true, y_pred, num_classes=10):
    y_true, y_pred = validate_input(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
    
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)    
    
    macro_f1_score = f1_score(y_true=y_true, y_pred=y_pred, average='macro', labels=range(num_classes), zero_division=0)
    
    return {"accuracy": accuracy, "macro_f1": macro_f1_score}

def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    y_true, y_pred = validate_input(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=range(num_classes))
    return cm

def compute_per_class_accuracy(y_true, y_pred, num_classes=10):
    cm = compute_confusion_matrix(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
    
    true_positive = np.diag(cm) 
    total_per_class = np.sum(cm, axis=1)
    return np.divide(true_positive, total_per_class, 
                     out=np.zeros(num_classes, dtype=float), 
                     where=total_per_class != 0)
    
    
def compute_per_class_metrics(y_true, y_pred, num_classes=10):
    y_true, y_pred = validate_input(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
    
    precision, recall, f1, support = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, labels=range(num_classes), zero_division=0)
    return {
        'precision': precision,
        "recall": recall,
        "f1": f1,
        "support": support # number of occurences of each label in y_true
    }    