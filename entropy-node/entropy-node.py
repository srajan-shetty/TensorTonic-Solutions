import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    y = np.unique(y, return_counts=True)[1]/len(y)

    return -1* np.sum(y*np.log2(y)).item()