from typing import Callable, Optional

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from pywhy_stats.kernel_utils import _default_regularization


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
    func: Callable,
    e_hat: ArrayLike,
    X: ArrayLike,
    Y: ArrayLike,
    null_reps: int = 1000,
    n_jobs=None,
    seed=None,
    **kwargs,
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
            delayed(func)(X, Y, rng.binomial(1, e_hat, size=n_samps), **kwargs)
            for _ in range(null_reps)
        ]
    )
    return null_dist
