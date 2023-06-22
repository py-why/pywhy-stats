from typing import Callable, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from pywhy_stats.kernel_utils import (
    _default_regularization,
    _get_default_kernel,
    _preprocess_kernel_data,
    compute_kernel,
)
from pywhy_stats.pvalue_result import PValueResult


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
    :footcite:`Zhang2011` that are computationally efficient.

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
    test_statistic, pvalue = _kernel_test(
        X,
        Y,
        group_ind,
        kernel_X,
        kernel_Y,
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
    Y: ArrayLike,
    group_ind: ArrayLike,
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

    # compute kernels in each data space
    if kernel_X is None:
        kernel_X = _get_default_kernel(X)
    if kernel_Y is None:
        kernel_Y = _get_default_kernel(Y)

    L = compute_kernel(
        Y,
        metric=kernel_Y,
        centered=centered,
        n_jobs=n_jobs,
    )
    K = compute_kernel(
        X,
        metric=kernel_X,
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
        e_hat, X=K, Y=L, null_reps=null_sample_size, seed=random_seed, n_jobs=n_jobs
    )

    # compute the pvalue
    pvalue = (1 + np.sum(null_dist >= stat)) / (1 + null_sample_size)
    return stat, pvalue


def _compute_propensity_scores(
    group_ind: ArrayLike,
    propensity_model: Optional[BaseEstimator] = None,
    propensity_weights: Optional[ArrayLike] = None,
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    **kwargs,
):
    if propensity_model is None:
        K: ArrayLike = kwargs.get("K")

        # compute a default penalty term if using a kernel matrix
        if K.shape[0] == K.shape[1]:
            propensity_penalty_ = _default_regularization(K)
            C = 1 / (2 * propensity_penalty_)
        else:
            propensity_penalty_ = 0.0
            C = 1.0

        # default model is logistic regression
        propensity_model_ = LogisticRegression(
            penalty="l2",
            n_jobs=n_jobs,
            warm_start=True,
            solver="lbfgs",
            random_state=random_state,
            C=C,
        )
    else:
        propensity_model_ = propensity_model

    # either use pre-defined propensity weights, or estimate them
    if propensity_weights is None:
        K = kwargs.get("K")
        # fit and then obtain the probabilities of treatment
        # for each sample (i.e. the propensity scores)
        propensity_weights = propensity_model_.fit(K, group_ind).predict_proba(K)[:, 1]
    else:
        propensity_weights = propensity_weights[:, 1]
    return propensity_weights


def compute_null(
    e_hat: ArrayLike, X: ArrayLike, Y: ArrayLike, null_reps: int = 1000, n_jobs=None, seed=None
) -> ArrayLike:
    """Estimate null distribution using propensity weights.

    Parameters
    ----------
    e_hat : Array-like of shape (n_samples,)
        The predicted propensity score for ``group_ind == 1``.
    X : Array-Like of shape (n_samples, n_features_x)
        The X (covariates) array.
    Y : Array-Like of shape (n_samples, n_features_y)
        The Y (outcomes) array.
    null_reps : int, optional
        Number of times to sample null, by default 1000.
    n_jobs : int, optional
        Number of jobs to run in parallel, by default None.
    seed : int, optional
        Random generator, or random seed, by default None.

    Returns
    -------
    null_dist : Array-like of shape (n_samples,)
        The null distribution of test statistics.
    """
    rng = np.random.default_rng(seed)

    n_samps = X.shape[0]

    # compute the test statistic on the conditionally permuted
    # dataset, where each group label is resampled for each sample
    # according to its propensity score
    null_dist = Parallel(n_jobs=n_jobs)(
        [
            delayed(_compute_test_statistic)(X, Y, rng.binomial(1, e_hat, size=n_samps))
            for _ in range(null_reps)
        ]
    )
    return null_dist


def _compute_test_statistic(K: ArrayLike, L: ArrayLike, group_ind: ArrayLike):
    n_samples = len(K)

    # compute W matrices from K and z
    W0, W1 = _compute_inverse_kernel(K, group_ind)

    # compute L kernels
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
