import numpy as np
from numpy._typing import ArrayLike
from scipy.stats import iqr
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import check_pairwise_arrays


def delta_kernel(X: ArrayLike, Y=None) -> ArrayLike:
    """Delta kernel for categorical values.

    This is, the similarity is 1 if the values are equal and 0 otherwise.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples_x, n_dimensions)
        Input data.
    Y : ArrayLike of shape (n_samples_y, n_dimensions), optional
       By default None.

    Returns
    -------
    result : ArrayLike of shape (n_samples, n_samples)
        The resulting kernel matrix after applying the delta kernel.
    """
    X, Y = check_pairwise_arrays(X, Y, dtype=str)

    return np.char.equal(X[:, np.newaxis], Y).all(axis=-1).astype(int)


def estimate_squared_sigma_rbf(
    X: ArrayLike,
    method="median",
    distance_metric: str = "euclidean",
) -> float:
    """
    Estimates the sigma**2 in K(x, x') = exp(-||x - x'|| / (2 * sigma**2)).

    Parameters
    ----------
    X : ArrayLike of shape (n_samples, n_features)
        The data.
    method : str, optional, {"median", "silverman"}
        Method to use, by default "median".
    distance_metric : str
        Only relevant for the "median" method. The distance metric to compute distances
        among samples within each data matrix, by default 'euclidean'. Can be any valid string
        as defined in :func:`sklearn.metrics.pairwise_distances`.

    Returns
    -------
    sigma_squared : float
        The estimated sigma**2 in K(x, x') = exp(-||x - x'|| / (2 * sigma**2)).
    """
    if method == "silverman":
        if X.ndim > 1 and X.shape[1] > 1:
            raise ValueError(
                "The Silverman method to estimate the kernel bandwidth is currently only "
                "supported for one dimensional data!"
            )
        X = X.reshape(-1)

        # https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator
        return 1 / (0.9 * np.min([np.std(X), iqr(X) / 1.34]) * (X.shape[0] ** (-1 / 5))) ** 2
    elif method == "median":
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Note: sigma = 1 / np.sqrt(kwidth)
        # compute N x N pairwise distance matrix
        dists = pairwise_distances(X, metric=distance_metric)

        # compute median of off diagonal elements
        med = np.median(dists[dists > 0])

        # prevents division by zero when used on label vectors
        med = med if med else 1

        return 1 / med**2
    else:
        raise NotImplementedError("Unknown kernel width estimation method %s!" % method)
