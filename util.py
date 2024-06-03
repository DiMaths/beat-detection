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

def sliding_min(x: np.ndarray, w: int) -> np.ndarray:
    """
    Computes sliding min (min in sliding window)
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
        y = np.minimum(y[step:], y[:-step])
        current_w += step
    return y


def relative_spikes(x: np.ndarray, w: int, min_rel_jump: float, debug_plot: bool = False) -> np.ndarray:

    """
    Computes local maximas which is relatively high enough to be spikes
    @param x: np.ndarray - original sequence
    @param w: int - size of the sliding window (size of the max/min neighbourhoods)
    @param min_rel_jump: float 
        - non-negative threshold for ratio (local_max - local_min) / local_min

    @returns spikes: np.ndarray - array of the same size as x
    """

    # normalize to [0,1] scale 
    x -= np.min(x[x > 0])
    x /= np.max(x)
    x[x < 0] = 0
    slide_max = sliding_max(x, w)
    slide_min = sliding_min(x, w)
    relative_jumps = np.array([(slide_max[i] / slide_min[i]) - 1 if slide_min[i] > 0 else float('inf') for i in range(len(x))], dtype=float)
    if debug_plot:
        import matplotlib.pyplot as plt
        plt.plot(relative_jumps, 'r')
        plt.hlines(min_rel_jump, 0, len(relative_jumps), 'b')
        plt.yscale('log')
        plt.show()

    local_maximas = np.where(slide_max == x)[0]
    possible_jumps = np.where(relative_jumps > min_rel_jump)[0]
    spikes = np.intersect1d(local_maximas, possible_jumps)
    if spikes[0] == 0:
        spikes = spikes[1:]
    return spikes