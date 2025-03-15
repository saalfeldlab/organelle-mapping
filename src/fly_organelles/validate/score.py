import torch
import zarr
import numpy as np

def balanced_accuracy(pred, label, eps=1e-7):
    """
    pred:  (B, C, D, H, W) or (B, C, X, Y) etc., thresholded/binary (0 or 1)
    label: (B, C, D, H, W) same shape, binary ground truth (0 or 1)
    
    Returns:
        A list (or 1D tensor) of length C with Balanced Accuracy for each channel.
    """
    # Ensure pred and label are the same shape
    assert pred.shape == label.shape
    
    B, C = pred.shape[:2]
    # Flatten spatial dimensions so we can sum easily:
    # new shape = (B, C, rest_of_dims...)
    # We'll sum over everything except B, C
    pred_flat = pred.view(B, C, -1)
    label_flat = label.view(B, C, -1)
    
    balanced_acc_scores = []
    
    for c in range(C):
        # For channel c, across the batch and spatial dims
        p = pred_flat[:, c]  # shape (B, num_voxels_per_batch)
        l = label_flat[:, c]
        
        # Sum across batch + spatial
        TP = torch.sum((p == 1) & (l == 1), dtype=torch.float)
        TN = torch.sum((p == 0) & (l == 0), dtype=torch.float)
        FP = torch.sum((p == 1) & (l == 0), dtype=torch.float)
        FN = torch.sum((p == 0) & (l == 1), dtype=torch.float)
        
        # Compute TPR (Sensitivity) and TNR (Specificity)
        TPR = TP / (TP + FN + eps)
        TNR = TN / (TN + FP + eps)
        
        # Balanced accuracy
        bal_acc = 0.5 * (TPR + TNR)
        balanced_acc_scores.append(bal_acc.item())
    
    return balanced_acc_scores

def validate_snapshot(path, score_function,score_function_kwargs={}, activation_function=None, threshold=0.5):
    z = zarr.open(path, mode='a')
    pred = z["output"][:]
    label = z["labels"][:]
    pred = torch.from_numpy(pred)
    label = torch.from_numpy(label)
    if activation_function is not None:
        pred = activation_function(pred)
    if threshold is not None:
        pred = (pred > threshold).float()
    if score_function_kwargs is not None:
        scores = score_function(pred, label, **score_function_kwargs)
    else:
        scores = score_function(pred, label)

    z.attrs["scores"] = scores
    
    return scores

def validate_snapshots(paths, score_function,score_function_kwargs={}, activation_function=None, threshold=0.5):
    scores = []
    for path in paths:
        validate_snapshot(path, score_function,score_function_kwargs=score_function_kwargs, activation_function=activation_function, threshold=threshold)
