from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.linalg import logm
from scipy.optimize import minimize_scalar
from sklearn.metrics import pairwise_kernels
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS

from pywhy_stats.kernels import delta_kernel, estimate_squared_sigma_rbf
from pywhy_stats.utils import TemporarilySetKey


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
    kernel: Optional[Union[Callable, str]] = None,
    centered: bool = True,
    n_jobs: Optional[int] = None,
) -> Tuple[ArrayLike, float]:
    """Compute a kernel matrix.

    If no kernel is specified, chooses a data appropriate kernel and estimates the kernel width parameter.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples_X, n_features_X)
        The X array.
    Y : ArrayLike of shape (n_samples_Y, n_features_Y), optional
        The Y array, by default None.
    kernel : str, optional
        Either a callable that computes a kernel matrix or a metric string as defined in
        :func:`sklearn.metrics.pairwise.pairwise_kernels`. Note 'rbf'
        and 'gaussian' are the same metric. If None is given, the kernel is inferred based on the data type, which is
        currently either the rbf kernel for continuous data or the delta kernel for categorical (string) data.
    centered : bool, optional
        Whether to center the kernel matrix or not, by default True.
        When centered, the kernel matrix induces a zero mean. The main purpose of
        centering is to remove the bias or mean shift from the data represented
        by the kernel matrix.
    n_jobs : int, optional
        The number of jobs to run computations in parallel, by default None. This only works for string kernel inputs.

    Returns
    -------
    kernel : ArrayLike of shape (n_samples_X, n_samples_X) or (n_samples_X, n_samples_Y)
        The kernel matrix.

    Notes
    -----
    If the kernel is a callable, it will have either one input for ``X``, or two inputs for ``X`` and
    ``Y``. It is assumed that the kernel operates on the entire array to compute
    the kernel array, that is, the callable performs a vectorized operation to compute the entire kernel matrix.

    If one has an unvectorizable kernel function, then it is advised to use the callable with the
    :func:`~sklearn.metrics.pairwise.pairwise_kernels` function, which will parallelize the kernel computation
    across each row of the data.
    """
    # Note that this is added to the list of possible kernels for :func:`~sklearn.metrics.pairwise.pairwise_kernels`.
    # because it is more efficient to compute the kernel over the entire matrices at once
    # since numpy has vectorized operations.
    # with temporarily_restrict_key("delta", PAIRWISE_KERNEL_FUNCTIONS):
    with TemporarilySetKey(PAIRWISE_KERNEL_FUNCTIONS, "delta", delta_kernel):
        PAIRWISE_KERNEL_FUNCTIONS["delta"] = delta_kernel

        # if the width of the kernel is not set, then use the median trick to set the
        # kernel width based on the data X
        if kernel is None:
            metric, kernel_params = _get_default_kernel(X)
            kernel_matrix = pairwise_kernels(
                X, Y=Y, metric=metric, n_jobs=n_jobs, filter_params=False, **kernel_params
            )
        elif isinstance(kernel, str):
            kernel_matrix = pairwise_kernels(
                X, Y=Y, metric=kernel, n_jobs=n_jobs, filter_params=False
            )
        else:
            # kernel is a callable
            if Y is None:
                kernel_matrix = kernel(X)
            else:
                kernel_matrix = kernel(X, Y)

        if centered:
            kernel_matrix = _fast_centering(kernel_matrix)

    return kernel_matrix


def correntropy_matrix(
    data: ArrayLike,
    kernel: Optional[Union[str, Callable[[ArrayLike], ArrayLike]]] = None,
    centered: bool = True,
    n_jobs: Optional[int] = None,
) -> ArrayLike:
    r"""Compute the centered correntropy of a matrix.

    Parameters
    ----------
    data : ArrayLike of shape (n_samples, n_features)
        The data.
    kernel : str, Callable, optional
        The kernel.
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
                kernel=kernel,
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

    if normalize_data:
        # first normalize the data to have zero mean and unit variance
        # along the columns of the data
        X = _normalize_data(X)
        Y = _normalize_data(Y)
        if Z is not None:
            Z = _normalize_data(Z)
    return X, Y, Z
