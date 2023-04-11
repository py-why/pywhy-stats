from math import log, sqrt
from typing import Optional, Set, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from .p_value_result import PValueResult


def fisherz(
    data: ArrayLike,
    x: int,
    y: int,
    sep_set: Optional[Set[int]] = None,
    correlation_matrix: Optional[ArrayLike] = None,
):
    """Perform an independence test using Fisher-Z's test.

    Works on Gaussian random variables.

    Parameters
    ----------
    data : ArrayLike of shape (n_samples, n_variables)
        The data.
    x : int
        The column index of the first node variable.
    y : int
        The column index the second node variable.
    sep_set : set of int
        The set of column nodes of x and y (as a set()). If `None` (default),
        will run a marginal independence test.
    correlation_matrix : np.ndarray of shape (n_variables, n_variables), optional
        ``None`` means without the parameter of correlation matrix and
        the correlation will be computed from the data., by default None

    Returns
    -------
    X : float
        The test statistic.
    p : float
        The p-value of the test.
    """
    if correlation_matrix is None:
        correlation_matrix = np.corrcoef(data.T)
    if sep_set is None:
        sep_set = set()
    sample_size = data.shape[0]
    var_idx = list({x, y}.union(sep_set))  # type: ignore

    # compute the correlation matrix within the specified data
    sub_corr_matrix = correlation_matrix[np.ix_(var_idx, var_idx)]
    inv = np.linalg.inv(sub_corr_matrix)
    r = -inv[0, 1] / sqrt(inv[0, 0] * inv[1, 1])

    # apply the Fisher Z-transformation
    Z = 0.5 * log((1 + r) / (1 - r))

    # compute the test statistic
    X = sqrt(sample_size - len(sep_set) - 3) * abs(Z)
    p = 2 * (1 - norm.cdf(abs(X)))
    return PValueResult(X, p)
