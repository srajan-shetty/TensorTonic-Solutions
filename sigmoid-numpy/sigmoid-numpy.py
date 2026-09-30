import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    x = np.asarray(x, dtype=float)
    out = 1 / (1 + np.exp(-x))
    return out.item() if out.ndim == 0 else out