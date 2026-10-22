import numpy as np
def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    # Write code here
    TP = np.intersect1d(np.array(recommended[:k]), np.array(relevant)).size
    n_relevant = len(relevant)
    return [TP/k,TP/n_relevant]