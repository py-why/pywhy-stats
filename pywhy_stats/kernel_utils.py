import inspect
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.linalg import logm
from scipy.optimize import minimize_scalar
from sklearn.metrics import pairwise_kernels
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS
from sklearn.preprocessing import LabelEncoder

from pywhy_stats.kernels import delta_kernel, estimate_squared_sigma_rbf


def _default_regularization(K: ArrayLike) -> float:
    """Compute a default regularization for Kernel Logistic Regression.

    Parameters
    ----------
    K : ArrayLike of shape (n_samples, n_samples)
        The kernel matrix.

    Returns
    -------
    x : float
        The default l2 regularization term.
    """
    n_samples = K.shape[0]
    svals = np.linalg.svd(K, compute_uv=False, hermitian=True)
    res = minimize_scalar(
        lambda reg: np.sum(svals**2 / (svals + reg) ** 2) / n_samples + reg,
        bounds=(0.0001, 1000),
        method="bounded",
    )
    return res.x


def _fast_centering(k: ArrayLike) -> ArrayLike:
    """
    Compute centered kernel matrix in time O(n^2).

    The centered kernel matrix is defined as K_c = H @ K @ H, with
    H = identity - 1/ n * ones(n,n). Computing H @ K @ H via matrix multiplication scales with n^3.
    The implementation circumvents this and runs in time n^2.

    Originally authored by Jonas Kuebler
    """
    n = len(k)
    return (
        k
        - 1 / n * np.outer(np.ones(n), np.sum(k, axis=0))
        - 1 / n * np.outer(np.sum(k, axis=1), np.ones(n))
        + 1 / n**2 * np.sum(k) * np.ones((n, n))
    )


def _get_default_kernel(X: ArrayLike) -> Tuple[str, Dict]:
    """Attempt to infer the kernel function from the data type of X.

    Parameters
    ----------
    X : ArrayLike
        Data array. If the data type is not string, the RBF kernel is used.
        If data type is string, the delta kernel is used.

    Returns
    -------
    Callable[[ArrayLike], ArrayLike]
        The kernel function.
    """
    if X.dtype.type not in (np.str_, np.object_):
        return "rbf", {"gamma": 0.5 * estimate_squared_sigma_rbf(X)}
    else:
        return "delta", dict()


def _normalize_data(X: ArrayLike) -> ArrayLike:
    """Normalize data to zero mean and unit variance."""
    for column in range(X.shape[1]):
        if isinstance(X[0, column], int) or isinstance(X[0, column], float):
            X[:, column] = stats.zscore(X[:, column])
            X[:, column] = np.nan_to_num(X[:, column])  # in case some dim of X is constant
    return X


def compute_kernel(
    X: ArrayLike,
    Y: Optional[ArrayLike] = None,
    metric: Optional[Union[Callable, str]] = None,
    centered: bool = True,
    n_jobs: Optional[int] = None,
) -> Tuple[ArrayLike, float]:
    """Compute a kernel matrix and corresponding width.

    Also optionally estimates the kernel width parameter.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples_X, n_features_X)
        The X array.
    Y : ArrayLike of shape (n_samples_Y, n_features_Y), optional
        The Y array, by default None.
    metric : str, optional
        The metric to compute the kernel function, by default 'rbf'.
        Can be any string as defined in
        :func:`sklearn.metrics.pairwise.pairwise_kernels`. Note 'rbf'
        and 'gaussian' are the same metric.
    centered : bool, optional
        Whether to center the kernel matrix or not, by default True.
        When centered, the kernel matrix induces a zero mean. The main purpose of
        centering is to remove the bias or mean shift from the data represented
        by the kernel matrix.
    n_jobs : int, optional
        The number of jobs to run computations in parallel, by default None.

    Returns
    -------
    kernel : ArrayLike of shape (n_samples_X, n_samples_X) or (n_samples_X, n_samples_Y)
        The kernel matrix.

    Notes
    -----
    If the metric is a callable, it will have either one input for ``X``, or two inputs for ``X`` and
    ``Y``. If one input is passed in, it is assumed that the kernel operates on the entire array to compute
    the kernel array. If two inputs are passed in, then it is assumed that the kernel operates on
    pairwise row vectors from each input array. This callable is parallelized across rows of the input
    using :func:`~sklearn.metrics.pairwise.pairwise_kernels`. Note that ``(X, Y=None)`` is a valid input
    signature for the kernel function and would then get passed to the pairwise kernel function. If
    a callable is passed in, it is generally faster and more efficient if one can define a vectorized
    operation that operates on the whole array at once. Otherwise, the pairwise kernel function will
    call the function for each combination of rows in the input arrays.
    """
    # Note that this is added to the list of possible kernels for :func:`~sklearn.metrics.pairwise.pairwise_kernels`.
    # because it is more efficient to compute the kernel over the entire matrices at once
    # since numpy has vectorized operations.
    PAIRWISE_KERNEL_FUNCTIONS["delta"] = delta_kernel

    # if the width of the kernel is not set, then use the median trick to set the
    # kernel width based on the data X
    if metric is None:
        metric, kernel_params = _get_default_kernel(X)
    else:
        kernel_params = dict()

    # if the metric is a callable, then we need to check the number of arguments
    # it takes. If it takes just one argument, then we bypass the pairwise kernel
    # and call the kernel function directly on the entire input array. See kci.py Notes
    # for more information.
    if callable(metric) and len(inspect.getfullargspec(metric).args) == 1:
        if Y is not None:
            raise RuntimeError("Y is not allowed when metric is a callable with one argument.")

        # If the number of arguments is just one, then we bypass the pairwise kernel
        # optimized computation via sklearn and opt to use the metric function directly
        kernel = metric(X)
    else:
        kernel = pairwise_kernels(
            X, Y=Y, metric=metric, n_jobs=n_jobs, filter_params=False, **kernel_params
        )

    if centered:
        kernel = _fast_centering(kernel)
    return kernel


def corrent_matrix(
    data: ArrayLike,
    metric: Optional[Union[str, Callable[[ArrayLike], ArrayLike]]] = None,
    centered: bool = True,
    n_jobs: Optional[int] = None,
) -> ArrayLike:
    r"""Compute the centered correntropy of a matrix.

    Parameters
    ----------
    data : ArrayLike of shape (n_samples, n_features)
        The data.
    metric : str
        The kernel metric.
    centered : bool, optional
        Whether to center the kernel matrix or not, by default True.
    n_jobs : int, optional
        The number of jobs to run computations in parallel, by default None.

    Returns
    -------
    data : ArrayLike of shape (n_features, n_features)
        A symmetric centered correntropy matrix of the data.

    Notes
    -----
    The estimator for the correntropy array is given by the formula
    :math:`1 / N \\sum_{i=1}^N k(x_i, y_i) - 1 / N**2 \\sum_{i=1}^N \\sum_{j=1}^N k(x_i, y_j)`.
    The first term is the estimate, and the second term is the bias, and together they form
    an unbiased estimate.
    """
    n_samples, n_features = data.shape
    corren_arr = np.zeros(shape=(n_features, n_features))

    # compute kernel between each feature, which is now (n_features, n_features) array
    for idx in range(n_features):
        for jdx in range(idx + 1):
            K = compute_kernel(
                X=data[:, [idx]],
                Y=data[:, [jdx]],
                metric=metric,
                centered=centered,
                n_jobs=n_jobs,
            )

            # compute the bias due to finite-samples
            bias = np.sum(K) / n_samples**2

            # compute the sample centered correntropy
            corren = (1.0 / n_samples) * np.trace(K) - bias

            corren_arr[idx, jdx] = corren_arr[jdx, idx] = corren
    return corren_arr


def von_neumann_divergence(A: ArrayLike, B: ArrayLike) -> float:
    """Compute Von Neumann divergence between two PSD matrices.

    Parameters
    ----------
    A : ArrayLike of shape (n_features, n_features)
        The first PSD matrix.
    B : ArrayLike of shape (n_features, n_features)
        The second PSD matrix.

    Returns
    -------
    div : float
        The divergence value.

    Notes
    -----
    The Von Neumann divergence, or what is known as the Bregman divergence in
    :footcite:`Yu2020Bregman` is computed as follows with
    :math:`D(A || B) = Tr(A (log(A) - log(B)) - A + B)`.
    """
    div = np.trace(A.dot(logm(A) - logm(B)) - A + B)
    return div


def _preprocess_kernel_data(
    X: ArrayLike,
    Y: ArrayLike,
    Z: Optional[ArrayLike] = None,
    normalize_data: bool = True,
):
    """Preprocess the data for kernel methods.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples_X, n_features_X)
        The X array.
    Y : ArrayLike of shape (n_samples_Y, n_features_Y)
        The Y array.
    Z : ArrayLike of shape (n_samples_Z, n_features_Z), optional
        The Z array, by default None.
    normalize_data : bool
        Whether to standard-normalize the data or not.
    """
    X = np.array(X)
    Y = np.array(Y)
    if Z is not None:
        Z = np.array(Z)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Z is not None and Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    # handle strings as categorical data automatically
    if X.dtype.type in (np.str_, np.object_, np.unicode_):
        enc = LabelEncoder()
        for idx in range(X.shape[1]):
            X[:, idx] = enc.fit_transform(X[:, idx])
    if Y.dtype.type in (np.str_, np.object_, np.unicode_):
        enc = LabelEncoder()
        for idx in range(Y.shape[1]):
            Y[:, idx] = enc.fit_transform(Y[:, idx])
    if Z is not None and Z.dtype.type in (np.str_, np.object_, np.unicode_):
        enc = LabelEncoder()
        for idx in range(Z.shape[1]):
            Z[:, idx] = enc.fit_transform(Z[:, idx])

    if normalize_data:
        # first normalize the data to have zero mean and unit variance
        # along the columns of the data
        X = _normalize_data(X)
        Y = _normalize_data(Y)
        if Z is not None:
            Z = _normalize_data(Z)
    return X, Y, Z
