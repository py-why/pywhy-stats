from typing import Union

import numpy as np
from numpy import exp, median, shape, sqrt, ndarray
from numpy.random import permutation
from scipy.spatial.distance import squareform, pdist


def rbf_kernel(X: np.ndarray, width: Union[float, str] = "median") -> np.ndarray:
    """
    Custom implementation of the RBF kernel.

    Parameters
    ----------
    X: numpy array of shape (n_samples, n_columns)
    width : string or float
            The width for the RBF kernel. This can either be provided as a float or as a string indicating one of
            the following approximation methods:
                'empirical_hsic': using empirical rule for HSIC
                'empirical_kci': using empirical rule for KCI
                'median': using the median trick

    Returns
    -------
    result: numpy array of shape (n_samples, n_samples)
            The resulting kernel matrix after applying the RBF kernel.
    """
    if isinstance(width, str):
        if width == "median":
            width = _get_gaussian_median_width(X)
        elif width == "empirical_hsic":
            width = _get_gaussian_empirical_hsic(X)
        elif width == "empirical_kci":
            width = _get_gaussian_empirical_kci(X)
        else:
            raise ValueError("Unknown width approximation method!")

    return exp(-0.5 * squareform(pdist(X, 'sqeuclidean')) * width)


def delta_kernel(X: np.ndarray) -> np.ndarray:
    """
    Delta kernel for categorical values. This is, the similarity is 1 if the values are equal and 0 otherwise.

    Parameters
    ----------
    X: numpy array of shape (n_samples, n_columns)

    Returns
    -------
    result: numpy array of shape (n_samples, n_samples)
            The resulting kernel matrix after applying the delta kernel.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    return np.array(list(map(lambda value: value == X, X))).reshape(X.shape[0], X.shape[0]).astype(np.float32)


def _get_gaussian_median_width(X: np.ndarray) -> float:
    n = shape(X)[0]
    if n > 1000:
        X = X[permutation(n)[:1000], :]
    dists = squareform(pdist(X, 'euclidean'))
    median_dist = median(dists[dists > 0])
    width = sqrt(2.) * median_dist
    return 1.0 / (width ** 2)


def _get_gaussian_empirical_kci(X: np.ndarray) -> float:
    n = shape(X)[0]
    if n < 200:
        width = 1.2
    elif n < 1200:
        width = 0.7
    else:
        width = 0.4
    theta = 1.0 / (width ** 2)
    return theta / X.shape[1]


def _get_gaussian_empirical_hsic(X: ndarray):
    n = shape(X)[0]
    if n < 200:
        width = 0.8
    elif n < 1200:
        width = 0.5
    else:
        width = 0.3
    theta = 1.0 / (width ** 2)
    return theta * X.shape[1]
