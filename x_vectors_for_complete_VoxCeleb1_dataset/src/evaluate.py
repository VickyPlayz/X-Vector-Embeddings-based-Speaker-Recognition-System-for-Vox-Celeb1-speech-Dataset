import numpy as np

def compute_eer(scores, labels):
    """
    scores: list of scores
    labels: list of 0 (non-target) or 1 (target)
    """
    scores = np.array(scores)
    labels = np.array(labels)
    
    fpr, tpr = [], []
    thresholds = np.sort(scores)
    
    # Naive naive implementation O(N^2) can be slow, 
    # Use sklearn roc_curve for efficiency if available, or optimized loop
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    
    fnr = 1 - tpr
    
    # EER is where FPR == FNR
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
    return eer, eer_threshold

def compute_min_dcf(scores, labels, p_target=0.01, c_miss=1, c_fa=1):
    # Simplified MinDCF calculation
    # ...
    return 0.0 # Placeholder
