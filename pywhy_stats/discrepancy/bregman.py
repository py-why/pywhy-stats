"""Bregman (conditional) discrepancy test.

Also known as a conditional k-sample test, where the null hypothesis is that the
conditional distributions are equal across different population groups. The
Bregman tests for conditional divergence using correntropy.

Returns
-------
PValueResult
    The result of the test, which includes the test statistic and pvalue.
"""

from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from pywhy_stats.kernel_utils import (
    _get_default_kernel,
    _preprocess_kernel_data,
    corrent_matrix,
    von_neumann_divergence,
)

from ..pvalue_result import PValueResult
from .base import _compute_propensity_scores, compute_null


def condind(
    X: ArrayLike,
    Y: ArrayLike,
    group_ind: ArrayLike,
    kernel: Optional[Callable[[ArrayLike], ArrayLike]] = None,
    null_sample_size: int = 1000,
    normalize_data: bool = True,
    propensity_model=None,
    propensity_weights=None,
    centered: bool = True,
    n_jobs: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> PValueResult:
    """
    Test whether Y conditioned on X is invariant across the groups.

    For testing conditional independence on continuous data, we compute Bregman divergences
    :footcite:`Yu2020Bregman`. This specifically
    tests the (conditional) invariance null hypothesis :math::

        P_{Z=1}(Y|X) = P_{Z=0}(Y|X)

    Parameters
    ----------
    X : ArrayLike of shape (n_samples, n_features_x)
        Data for variable X, which can be multidimensional.
    Y : ArrayLike of shape (n_samples, n_features_y)
        Data for variable Y, which can be multidimensional.
    group_ind : ArrayLike of shape (n_samples,)
        Data for group indicator Z, which can be multidimensional.
    kernel_X : Callable[[ArrayLike], ArrayLike]
        The kernel function for X. By default, the RBF kernel is used for continuous and the delta
        kernel for categorical data. Note that we currently only consider string values as categorical data.
    kernel_Y : Callable[[ArrayLike], ArrayLike]
        The kernel function for Y. By default, the RBF kernel is used for continuous and the delta
        kernel for categorical data. Note that we currently only consider string values as categorical data.
    null_sample_size : int
        The number of samples to generate for the bootstrap distribution to approximate the pvalue,
        by default 1000.
    normalize_data : bool
        Whether the data should be standardized to unit variance, by default True.
    propensity_model : Optional[BaseEstimator], optional
        The propensity model to use to estimate the propensity score, by default None.
    propensity_weights : Optional[ArrayLike], optional
        The propensity weights to use, by default None, which means that the propensity scores
        will be estimated from the propensity_model.
    centered : bool
        Whether the kernel matrix should be centered, by default True.
    n_jobs : Optional[int], optional
        The number of jobs to run in parallel, by default None.
    random_seed : Optional[int], optional
        Random seed, by default None.

    Notes
    -----
    Any callable can be given to create the kernel matrix. For instance, to use a particular kernel
    from sklearn::

        kernel_X = func:`sklearn.metrics.pairwise.pairwise_kernels.polynomial`

    References
    ----------
    .. footbibliography::
    """
    test_statistic, pvalue = _bregman_test(
        X=X,
        group_ind=group_ind,
        Y=Y,
        kernel=kernel,
        propensity_weights=propensity_weights,
        propensity_model=propensity_model,
        null_sample_size=null_sample_size,
        normalize_data=normalize_data,
        centered=centered,
        n_jobs=n_jobs,
        random_seed=random_seed,
    )
    return PValueResult(pvalue=pvalue, statistic=test_statistic)


def _bregman_test(
    X: ArrayLike,
    group_ind: ArrayLike,
    Y: ArrayLike,
    kernel: Optional[Callable[[np.ndarray], np.ndarray]],
    propensity_weights: Optional[ArrayLike],
    propensity_model: Optional[BaseEstimator],
    null_sample_size: int,
    normalize_data: bool,
    centered: bool,
    n_jobs: Optional[int],
    random_seed: Optional[int],
) -> Tuple[float, float]:
    X, Y, _ = _preprocess_kernel_data(X, Y, normalize_data=normalize_data)

    # compute kernels in each data space
    if kernel is None:
        kernel = _get_default_kernel(X)

    # We are interested in testing: P_1(y|x) = P_2(y|x)
    # compute the conditional divergence, which is symmetric by construction
    # 1/2 * (D(p_1(y|x) || p_2(y|x)) + D(p_2(y|x) || p_1(y|x)))
    conditional_div = _compute_test_statistic(
        X, Y, group_ind, metric=kernel, centered=centered, n_jobs=n_jobs
    )

    # compute propensity scores
    e_hat = _compute_propensity_scores(
        group_ind,
        propensity_model=propensity_model,
        propensity_weights=propensity_weights,
        n_jobs=n_jobs,
        random_state=random_seed,
        K=X,
    )

    # now compute null distribution
    # now compute null distribution
    null_dist = compute_null(
        _compute_test_statistic,
        e_hat,
        X=X,
        Y=Y,
        null_reps=null_sample_size,
        seed=random_seed,
        n_jobs=n_jobs,
        metric=kernel,
        centered=centered,
    )

    # compute pvalue
    pvalue = (1.0 + np.sum(null_dist >= conditional_div)) / (1 + null_sample_size)
    return conditional_div, pvalue


def _compute_test_statistic(
    X: ArrayLike,
    Y: ArrayLike,
    group_ind: ArrayLike,
    metric: Callable[[ArrayLike], ArrayLike],
    centered: bool = True,
    n_jobs: Optional[int] = None,
) -> float:
    first_group = group_ind == 0
    second_group = group_ind == 1
    X1 = X[first_group, :]
    X2 = X[second_group, :]
    Y1 = Y[first_group, :]
    Y2 = Y[second_group, :]

    # first compute the centered correntropy matrices, C_xy^1
    Cx1y1 = corrent_matrix(np.hstack((X1, Y1)), metric=metric, centered=centered, n_jobs=n_jobs)
    Cx2y2 = corrent_matrix(np.hstack((X2, Y2)), metric=metric, centered=centered, n_jobs=n_jobs)

    # compute the centered correntropy matrices for just C_x^1 and C_x^2
    Cx1 = corrent_matrix(
        X1,
        metric=metric,
        centered=centered,
        n_jobs=n_jobs,
    )
    Cx2 = corrent_matrix(
        X2,
        metric=metric,
        centered=centered,
        n_jobs=n_jobs,
    )

    # compute the conditional divergence with the Von Neumann div
    # D(p_1(y|x) || p_2(y|x))
    joint_div1 = von_neumann_divergence(Cx1y1, Cx2y2)
    joint_div2 = von_neumann_divergence(Cx2y2, Cx1y1)
    x_div1 = von_neumann_divergence(Cx1, Cx2)
    x_div2 = von_neumann_divergence(Cx2, Cx1)

    # compute the conditional divergence, which is symmetric by construction
    # 1/2 * (D(p_1(y|x) || p_2(y|x)) + D(p_2(y|x) || p_1(y|x)))
    conditional_div = 1.0 / 2 * (joint_div1 - x_div1 + joint_div2 - x_div2)
    return conditional_div
