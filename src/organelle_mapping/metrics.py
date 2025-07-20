from sklearn.metrics import accuracy_score, jaccard_score
from scipy.spatial.distance import dice
from skimage.measure import label
import numpy as np
from scipy.optimize import linear_sum_assignment


# def prepare_instance(y):
#     y = prepare_semantic(y)
#     y = label(y, connectivity=y.ndim)
#     return y


def dice_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Dice score between two binary images.
    """
    if np.sum(y_true + y_pred) == 0:
        # If both images are empty, return 1.0 (perfect match)
        return 1.0
    score = 1 - dice(y_true.flatten(), y_pred.flatten())
    if np.isnan(score):
        return 1.0
    return score

def jaccard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return jaccard_score(y_true.flatten(), y_pred.flatten(), zero_division=1.)


# def match_instance(y_pred, y_true):
#     pred_ids = np.unique(y_pred)
#     pred_ids = pred_ids[pred_ids != 0]
#     true_ids = np.unique(y_true)
#     true_ids = true_ids[true_ids != 0]
#     cost_matrix = np.zeros((len(true_ids), len(pred_ids)), dtype=np.float32)
#     y_pred = y_pred.flatten()
#     y_true = y_true.flatten()
#     matched_y_pred = np.zeros_like(y_pred)
#     for pid in pred_ids:
#         pid_mask = y_pred == pid
#         relevant_tids = np.unique(y_true[pid_mask])
#         relevant_tids = relevant_tids[relevant_tids != 0]
#         for tid in relevant_tids:
#             tid_mask = y_true == tid
#             pidid = pred_ids.tolist().index(pid)
#             tidid = true_ids.tolist().index(tid)
#             cost_matrix[tidid, pidid] = jaccard(pid_mask, tid_mask)

#     row_inds, col_inds = linear_sum_assignment(cost_matrix, maximize=True)
#     for pidid,tidid in zip(col_inds, row_inds):
#         pid = pred_ids[pidid]
#         tid = true_ids[tidid]
#         pid_mask = y_pred == pid
#         matched_y_pred[pid_mask] = tid
#     return matched_y_pred

# def accuracy(y_pred, y_true):
#     """
#     Compute the accuracy between two binary images.
#     """
#     y_pred = match_instance(y_pred, y_true)
#     return accuracy_score(y_pred.flatten(), y_true.flatten())
