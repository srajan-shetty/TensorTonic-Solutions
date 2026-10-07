import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype = float).ravel()
    n,d = X.shape
    
    W = np.zeros(d)
    b = 0.0

    for i in range(steps):
        p = _sigmoid(X@W + b)
        err = p-y
        W -=lr*(X.T)@(err)/n
        b -=lr*err.mean()

    return W , float(b)