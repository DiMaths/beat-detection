import numpy as np


def spectral_diff(X: np.ndarray, p_norm: int = 2, positive_only: bool = False) -> np.ndarray:
    """
    Computes spectral difference
    @param X: np.ndarray - original STFT
    @param p_norm: int - L_p norm is used
    @param positive_only: bool 
        - if True, cares only about positive components of the difference,
        i.e. only increments matter

    @returns diff: np.ndarray - norm(X[t] - X[t-1])
    """

    X_log = np.log10(1 + 10 * np.abs(X))
    X_delayed = np.zeros_like(X_log)
    X_delayed[:, 1:] = X_log[:, :-1].copy()
    diff = X_log - X_delayed

    if positive_only:
        diff = diff + np.abs(diff)
        diff /= 2
    diff = np.linalg.norm(diff, axis=0, ord=p_norm)

    return diff

