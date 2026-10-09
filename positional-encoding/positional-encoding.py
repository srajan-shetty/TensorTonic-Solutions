import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    pos = np.arange(seq_len).reshape((-1,1))
    div = base ** (np.arange(0,d_model,2)/d_model)
    pe= np.zeros((seq_len,d_model))
    pe[:,0::2] = np.sin(pos/div)
    pe[:,1::2] = np.cos(pos/div[:d_model//2])

    return pe
                