import numpy as np


def cosine_distance(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1 - ((Y @ X.T) / np.sqrt((Y ** 2).sum(axis=1).reshape((-1, 1)) @ (X ** 2).sum(axis=1).reshape((1, -1))))


def euclidean_distance(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    return ((Y ** 2).sum(axis=1)[:, None] + (X ** 2).sum(axis=1).T[None, :] - 2 * Y @ X.T) ** 0.5
