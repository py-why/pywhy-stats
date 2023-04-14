"""Independence test using Fisher-Z's test.

This test is also known as the partial correlation independence test.
It works on Gaussian random variables.

Parameters
----------
X : ArrayLike of shape (n_samples,)
    The first node variable.
Y : ArrayLike of shape (n_samples,)
    The second node variable.
condition_on : ArrayLike of shape (n_samples, n_variables)
    If `None` (default), will run a marginal independence test.
correlation_matrix : np.ndarray of shape (n_variables, n_variables), optional
    ``None`` means without the parameter of correlation matrix and
    the correlation will be computed from the data., by default None.

Notes
-----
When the data is not Gaussian, this test is not valid. In this case, we recommend
using the Kernel independence test at <insert link>.
"""

from math import log, sqrt
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from .p_value_result import PValueResult


def ind(X: ArrayLike, Y: ArrayLike, correlation_matrix: Optional[ArrayLike] = None) -> PValueResult:
    """Perform an independence test using Fisher-Z's test.

    Works on Gaussian random variables. This test is also known as the
    correlation test.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples,)
        The first node variable.
    Y : ArrayLike of shape (n_samples,)
        The second node variable.
    correlation_matrix : ArrayLike of shape (2, 2), optional
        The precomputed correlation matrix between X and Y., by default None.

    Returns
    -------
    X : float
        The test statistic.
    p : float
        The p-value of the test.
    """
    return _fisherz(X, Y, condition_on=None, correlation_matrix=correlation_matrix)


def condind(
    X: ArrayLike,
    Y: ArrayLike,
    condition_on: ArrayLike,
    correlation_matrix: Optional[ArrayLike] = None,
) -> PValueResult:
    """Perform an independence test using Fisher-Z's test.

    Works on Gaussian random variables. This test is also known as the
    correlation test.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples,)
        The first node variable.
    Y : ArrayLike of shape (n_samples,)
        The second node variable.
    condition_on : ArrayLike of shape (n_samples, n_variables)
        The conditioning set.
    correlation_matrix : ArrayLike of shape (2 + n_variables, 2 + n_variables), optional
        The precomputed correlation matrix between X, Y and ``condition_on``, by default None.

    Returns
    -------
    X : float
        The test statistic.
    p : float
        The p-value of the test.
    """
    return _fisherz(X, Y, condition_on=condition_on, correlation_matrix=correlation_matrix)


def _fisherz(
    X: ArrayLike,
    Y: ArrayLike,
    condition_on: Optional[ArrayLike] = None,
    correlation_matrix: Optional[ArrayLike] = None,
) -> PValueResult:
    """Perform an independence test using Fisher-Z's test.

    Works on Gaussian random variables. This test is also known as the
    partial correlation test.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples,)
        The first node variable.
    Y : ArrayLike of shape (n_samples,)
        The second node variable.
    condition_on : ArrayLike of shape (n_samples, n_variables)
        If `None` (default), will run a marginal independence test.
    correlation_matrix : np.ndarray of shape (n_variables, n_variables), optional
        ``None`` means without the parameter of correlation matrix and
        the correlation will be computed from the data., by default None.

    Returns
    -------
    X : float
        The test statistic.
    p : float
        The p-value of the test.
    """
    if condition_on is None:
        condition_on = np.empty((X.shape[0], 0))

    # compute the correlation matrix within the specified data
    data = np.hstack((X, Y, condition_on))
    sample_size = data.shape[0]
    if correlation_matrix is None:
        correlation_matrix = np.corrcoef(data.T)

    inv = np.linalg.pinv(correlation_matrix)
    r = -inv[0, 1] / sqrt(inv[0, 0] * inv[1, 1])

    # apply the Fisher Z-transformation
    Z = 0.5 * log((1 + r) / (1 - r))

    # compute the test statistic
    statistic = sqrt(sample_size - condition_on.shape[1] - 3) * abs(Z)
    p = 2 * (1 - norm.cdf(abs(statistic)))
    return PValueResult(statistic=statistic, pvalue=p)
