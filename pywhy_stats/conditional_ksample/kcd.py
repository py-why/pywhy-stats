"""Kernel (conditional) discrepancy test.

Also known as a conditional k-sample test, where the null hypothesis is that the
conditional distributions are equal across different population groups.

Returns
-------
PValueResult
    The result of the test, which includes the test statistic and pvalue.
"""

from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer

from pywhy_stats.kernel_utils import (
    _default_regularization,
    _preprocess_kernel_data,
    compute_kernel,
)

from ..pvalue_result import PValueResult
from .base_propensity import _compute_propensity_scores, _preprocess_propensity_data, compute_null


# XXX: determine if we can do this with Y being optional.
def condind(
    X: ArrayLike,
    Y: ArrayLike,
    group_ind: ArrayLike,
    kernel_X: Optional[Callable[[ArrayLike], ArrayLike]] = None,
    kernel_Y: Optional[Callable[[ArrayLike], ArrayLike]] = None,
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

    For testing conditional independence on continuous data, we leverage kernels
    :footcite:`Zhang2011` that are computationally efficient. This specifically
    tests the (conditional) invariance null hypothesis :math::

        P_{Z=1}(Y|X) = P_{Z=0}(Y|X)

    Parameters
    ----------
    X : ArrayLike of shape (n_samples, n_features_x)
        Data for variable X, which can be multidimensional.
    Y : ArrayLike of shape (n_samples, n_features_y)
        Data for variable Y, which can be multidimensional.
    group_ind : ArrayLike of shape (n_samples,)
        Data for group indicator Z, which can be multidimensional. This assigns each sample
        to a group indicated by 0 or 1.
    kernel_X : Callable[[ArrayLike], ArrayLike]
        The kernel function for X. By default, the RBF kernel is used for continuous and the delta
        kernel for categorical data. Note that we currently only consider string values as categorical data.
        Kernels can be specified in the same way as for :func:`~sklearn.metrics.pairwise.pairwise_kernels`
        with the addition that 'delta' kernel is supported for categorical data.
    kernel_Y : Callable[[ArrayLike], ArrayLike]
        The kernel function for Y. By default, the RBF kernel is used for continuous and the delta
        kernel for categorical data. Note that we currently only consider string values as categorical data.
        Kernels can be specified in the same way as for :func:`~sklearn.metrics.pairwise.pairwise_kernels`
        with the addition that 'delta' kernel is supported for categorical data.
    null_sample_size : int
        The number of samples to generate for the bootstrap distribution to approximate the pvalue,
        by default 1000.
    normalize_data : bool
        Whether the data should be standardized to unit variance, by default True.
    propensity_model : Optional[sklearn.base.BaseEstimator], optional
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

    In addition, we implement an efficient delta kernel. The delta kernel can be specified using the
    'kernel' string argument.

    References
    ----------
    .. footbibliography::
    """
    test_statistic, pvalue = _kernel_test(
        X=X,
        Y=Y,
        group_ind=group_ind,
        kernel_X=kernel_X,
        kernel_Y=kernel_Y,
        propensity_weights=propensity_weights,
        propensity_model=propensity_model,
        null_sample_size=null_sample_size,
        normalize_data=normalize_data,
        centered=centered,
        n_jobs=n_jobs,
        random_seed=random_seed,
    )
    return PValueResult(pvalue=pvalue, statistic=test_statistic)


def _kernel_test(
    X: ArrayLike,
    group_ind: ArrayLike,
    Y: ArrayLike,
    kernel_X: Optional[Callable[[np.ndarray], np.ndarray]],
    kernel_Y: Optional[Callable[[np.ndarray], np.ndarray]],
    propensity_weights: Optional[ArrayLike],
    propensity_model: Optional[BaseEstimator],
    null_sample_size: int,
    normalize_data: bool,
    centered: bool,
    n_jobs: Optional[int],
    random_seed: Optional[int],
) -> Tuple[float, float]:
    X, Y, _ = _preprocess_kernel_data(X, Y, normalize_data=normalize_data)
    _preprocess_propensity_data(
        group_ind=group_ind,
        propensity_weights=propensity_weights,
        propensity_model=propensity_model,
    )

    enc = LabelBinarizer(neg_label=0, pos_label=1)
    group_ind = enc.fit_transform(group_ind)

    # compute kernels in each data space
    L = compute_kernel(
        Y,
        kernel=kernel_Y,
        centered=centered,
        n_jobs=n_jobs,
    )
    K = compute_kernel(
        X,
        kernel=kernel_X,
        centered=centered,
        n_jobs=n_jobs,
    )

    # compute the test statistic
    stat = _compute_test_statistic(K, L, group_ind)

    # compute propensity scores
    e_hat = _compute_propensity_scores(
        group_ind,
        propensity_model=propensity_model,
        propensity_weights=propensity_weights,
        n_jobs=n_jobs,
        random_state=random_seed,
        K=K,
    )

    # now compute null distribution
    null_dist = compute_null(
        _compute_test_statistic,
        e_hat,
        X=K,
        Y=L,
        null_reps=null_sample_size,
        seed=random_seed,
        n_jobs=n_jobs,
    )

    # compute the pvalue
    pvalue = (1 + np.sum(null_dist >= stat)) / (1 + null_sample_size)
    return stat, pvalue


def _compute_test_statistic(K: ArrayLike, L: ArrayLike, group_ind: ArrayLike):
    n_samples = len(K)

    # compute W matrices from K and z
    W0, W1 = _compute_inverse_kernel(K, group_ind)

    # compute L kernels
    group_ind = np.squeeze(group_ind)
    first_mask = np.array(1 - group_ind, dtype=bool)
    second_mask = np.array(group_ind, dtype=bool)
    L0 = L[np.ix_(first_mask, first_mask)]
    L1 = L[np.ix_(second_mask, second_mask)]
    L01 = L[np.ix_(first_mask, second_mask)]

    # compute the final test statistic
    K0 = K[:, first_mask]
    K1 = K[:, second_mask]
    KW0 = K0 @ W0
    KW1 = K1 @ W1

    # compute the three terms in Lemma 4.4
    first_term = np.trace(KW0.T @ KW0 @ L0)
    second_term = np.trace(KW1.T @ KW0 @ L01)
    third_term = np.trace(KW1.T @ KW1 @ L1)

    # compute final statistic
    stat = (first_term - 2 * second_term + third_term) / n_samples
    return stat


def _compute_inverse_kernel(K, z) -> Tuple[ArrayLike, ArrayLike]:
    """Compute W matrices as done in KCD test.

    Parameters
    ----------
    K : ArrayLike of shape (n_samples, n_samples)
        The kernel matrix.
    z : ArrayLike of shape (n_samples)
        The indicator variable of 1's and 0's for which samples belong
        to which group.

    Returns
    -------
    W0 : ArrayLike of shape (n_samples_i, n_samples_i)
        The inverse of the kernel matrix from the first group.
    W1 : NDArraArrayLike of shape (n_samples_j, n_samples_j)
        The inverse of the kernel matrix from the second group.

    Notes
    -----
    Compute the W matrix for the estimated conditional average in
    the KCD test :footcite:`Park2021conditional`.

    References
    ----------
    .. footbibliography::
    """
    # compute kernel matrices
    z = np.squeeze(z)
    first_mask = np.array(1 - z, dtype=bool)
    second_mask = np.array(z, dtype=bool)

    K0 = K[np.ix_(first_mask, first_mask)]
    K1 = K[np.ix_(second_mask, second_mask)]

    # compute regularization factors
    regs_0 = _default_regularization(K0)
    regs_1 = _default_regularization(K1)

    # compute the number of samples in each
    n0 = int(np.sum(1 - z))
    n1 = int(np.sum(z))

    W0 = np.linalg.inv(K0 + regs_0 * np.identity(n0))
    W1 = np.linalg.inv(K1 + regs_1 * np.identity(n1))
    return W0, W1
