import numpy as np

def sliding_max(x: np.ndarray, w: int) -> np.ndarray:
    """
    Computes sliding max (max in sliding window)
    @param x: np.ndarray - original sequence
    @param w: int - size of the sliding window

    @returns y: np.ndarray - array of the same size as x
    """
    if w <= 0:
        raise ValueError(f"Window size must be positive integer, but got {w}.")

    # y is copy of x plus padded 0s
    y = np.zeros(x.shape[0] + w - 1)
    w_half = int(w/2)
    y[w_half:-(w-w_half-1)] = x  

    # recursive implementation allows O(len(x) * log(w)) complexity 
    # instead of O(len(x) * w)
    current_w = 1
    while current_w < w:
        step = min(current_w, w-current_w)
        y = np.maximum(y[step:], y[:-step])
        current_w += step
    return y