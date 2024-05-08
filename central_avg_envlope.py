import numpy as np

def moving_central_average(x: np.ndarray, w: int, mode: str = 'same', gaussian_smoothing: bool = False) -> np.ndarray:
    """
    Computes moving average of sequence x with window size w
    @param x: np.ndarray
    @param w: int - window size
    @param mode: str 
        - if 'same' then returns same length array by adding zeros to ends
        - if 'valid' return array of length (len(x) - w + 1)
    @param gaussian_smoothing: bool 
        - if True weights are taken from normal distribution pdf
    """
    if mode not in ["same", "valid"]:
        raise ValueError("Wrong mode of moving central averaging, allowed are 'same' and 'valid'.")
    if w <= 0:
        raise ValueError(f"Window size for moving central average must be positive integer, but got {w}.") 
    
    weights = np.ones(w)
    if gaussian_smoothing:
        eq_points = np.linspace(0, 1, num=w+2)[1:]
        weights = [(2*np.pi)**(-0.5) * np.exp(-0.5 * point**2) for point in eq_points]
        max_weight = max(weights)
        weights = [w/max_weight for w in weights]

    return np.convolve(x, weights, mode) / sum(weights)
