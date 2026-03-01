import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    """
    # ROC curve is all about finding the tradeoff between thetrue postive rate
    # and false positive rate **at various probability threshould**
    # 1. sort the predictions from highest to lowest
    # 2. iterate scores through as potential threshould 
    # 3. calculate TPR and FPR at each threshould 
    # TPR = TP/(TP+FN) of all actuall postivies, how many did we catch ?
    # FPR = FP/(FP+TN) of all actual negatives, how many did we wrongly flag ?

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # 1. Sort scores and the corresponding true labels in descending order
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # 2. Get unique thresholds (to avoid redundant calculations)
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # 3. Calculate True Positives and False Positives at each threshold
    # Accumulate the counts of 1s and 0s as we lower the threshold
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = np.cumsum(1 - y_true)[threshold_idxs]

    # 4. Convert counts to rates
    # TPR = TP / Total Positives; FPR = FP / Total Negatives
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]

    # Add (0,0) to the start of the curve
    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]
    thresholds = np.r_[np.inf, y_score[threshold_idxs]]

    return fpr, tpr, thresholds
    