import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if not seqs:
        return np.array(np.empty((0,0)), dtype=int)
    if max_len is None:
        max_len = max(len(i) for i in seqs)
        
    return np.array([(i+[pad_value]*(max_len-len(i)))[:max_len] for i in seqs])