"""Independence test using Kernel test.

Examples
--------
>>> import pywhy_stats as ps
>>> res = ps.kci.ind([1, 2, 3], [4, 5, 6])
>>> print(res.pvalue)
>>> 1.0
"""

from functools import partial
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from sklearn.metrics.pairwise import rbf_kernel

from pywhy_stats.kernels import delta_kernel, estimate_squared_sigma_rbf
from pywhy_stats.pvalue_result import PValueResult
from pywhy_stats.utils import preserve_random_state


@preserve_random_state
def ind(
    X: np.ndarray,
    Y: np.ndarray,
    kernel_X: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    kernel_Y: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    approx: bool = True,
    null_sample_size: int = 1000,
    threshold: float = 1e-5,
    normalize_data: bool = True,
    random_seed: Optional[int] = None,
) -> PValueResult:
    """
    Test whether X and Y are unconditionally independent using the Kernel Independence Tests.

    For testing independence on continuous data, we leverage kernels :footcite:`Zhang2011` that
    are computationally efficient.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples, n_features_x)
        Data for variable X, which can be multidimensional.
    Y : ArrayLike of shape (n_samples, n_features_y)
        Data for variable Y, which can be multidimensional.
    kernel_X : Callable[[ArrayLike], ArrayLike]
        The kernel function for X. By default, the RBF kernel is used for numeric and the delta
        kernel for categorical data. Note that we currently only consider string values as categorical data.
    kernel_Y : Callable[[ArrayLike], ArrayLike]
        The kernel function for Y. By default, the RBF kernel is used for continuous and the delta
        kernel for categorical data. Note that we currently only consider string values as categorical data.
    approx : bool
        Whether to use the Gamma distribution approximation for the pvalue, by default True.
    null_sample_size : int
        The number of samples to generate for the bootstrap distribution to approximate the pvalue,
        by default 1000.
    threshold : float
        The threshold set on the value of eigenvalues, by default 1e-5. Used to regularize the
        method.
    normalize_data : bool
        Whether the data should be standardized to unit variance, by default True.
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
        X, Y, None, kernel_X, kernel_Y, None, approx, null_sample_size, threshold, normalize_data
    )
    return PValueResult(pvalue=pvalue, statistic=test_statistic)


@preserve_random_state
def condind(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    kernel_X: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    kernel_Y: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    kernel_Z: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    approx: bool = True,
    null_sample_size: int = 1000,
    threshold: float = 1e-5,
    normalize_data: bool = True,
    random_seed: Optional[int] = None,
) -> PValueResult:
    """
    Test whether X and Y given Z are conditionally independent using Kernel Independence Tests.

    For testing conditional independence on continuous data, we leverage kernels
    :footcite:`Zhang2011` that are computationally efficient.

    Note that the conditional kernel test is significantly slower than the unconditional
    kernel test due to additional computational complexity by incorporating the
    conditioning variables.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples, n_features_x)
        Data for variable X, which can be multidimensional.
    Y : ArrayLike of shape (n_samples, n_features_y)
        Data for variable Y, which can be multidimensional.
    Z : ArrayLike of shape (n_samples, n_features_z)
        Data for variable Z, which can be multidimensional.
    kernel_X : Callable[[ArrayLike], ArrayLike]
        The kernel function for X. By default, the RBF kernel is used for continuous and the delta
        kernel for categorical data. Note that we currently only consider string values as categorical data.
    kernel_Y : Callable[[ArrayLike], ArrayLike]
        The kernel function for Y. By default, the RBF kernel is used for continuous and the delta
        kernel for categorical data. Note that we currently only consider string values as categorical data.
    kernel_Z : Callable[[ArrayLike], ArrayLike]
        The kernel function for Z. By default, the RBF kernel is used for continuous and the delta
        kernel for categorical data. Note that we currently only consider string values as categorical data.
    approx : bool
        Whether to use the Gamma distribution approximation for the pvalue, by default True.
    null_sample_size : int
        The number of samples to generate for the bootstrap distribution to approximate the pvalue,
        by default 1000.
    threshold : float
        The threshold set on the value of eigenvalues, by default 1e-5. Used to regularize the
        method.
    normalize_data : bool
        Whether the data should be standardized to unit variance, by default True.
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
        X, Y, Z, kernel_X, kernel_Y, kernel_Z, approx, null_sample_size, threshold, normalize_data
    )
    return PValueResult(pvalue=pvalue, statistic=test_statistic)


def _kernel_test(
    X: ArrayLike,
    Y: ArrayLike,
    Z: Optional[ArrayLike],
    kernel_X: Optional[Callable[[np.ndarray], np.ndarray]],
    kernel_Y: Optional[Callable[[np.ndarray], np.ndarray]],
    kernel_Z: Optional[Callable[[np.ndarray], np.ndarray]],
    approx: bool,
    null_sample_size: int,
    threshold: float,
    normalize_data: bool,
) -> Tuple[float, float]:
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

    if kernel_X is None:
        kernel_X = _get_default_kernel(X)
    if kernel_Y is None:
        kernel_Y = _get_default_kernel(Y)

    Kx = kernel_X(X)
    Ky = kernel_Y(Y)
    if Z is not None:
        if kernel_Z is None:
            kernel_Z = _get_default_kernel(Z)
        Kz = kernel_Z(Z)
        # Equivalent to concatenating them beforehand.
        # However, here we can then have individual kernels.
        Kx = Kx * Kz
        Kz = _fast_centering(Kz)

    Kx = _fast_centering(Kx)
    Ky = _fast_centering(Ky)

    if Z is None:
        return _ind(Kx, Ky, approx, null_sample_size, threshold)
    else:
        return _cond(Kx, Ky, Kz, approx, null_sample_size, threshold)


def _ind(Kx, Ky, approx, null_sample_size, threshold):
    # test statistic is the normal bivariate independence test
    test_stat = _compute_V_statistic(Kx, Ky)

    if approx:
        # approximate the pvalue using the Gamma distribution
        k_appr, theta_appr = _approx_gamma_params_ind(Kx, Ky)
        pvalue = 1 - stats.gamma.cdf(test_stat, k_appr, 0, theta_appr)
    else:
        null_samples = _compute_null_ind(Kx, Ky, null_sample_size, 1000, threshold)
        pvalue = np.sum(null_samples > test_stat) / float(null_sample_size)

    return test_stat, pvalue


def _cond(Kx, Ky, Kz, approx, null_sample_size, threshold):
    # compute the centralizing matrix for the kernels according to
    # conditioning set Z
    epsilon = 1e-6
    n = Kx.shape[0]
    Rz = epsilon * np.linalg.pinv(Kz + epsilon * np.eye(n))

    # compute the centralized kernel matrices
    KxzR = Rz.dot(Kx).dot(Rz)
    KyzR = Rz.dot(Ky).dot(Rz)

    # compute the conditional independence test statistic
    test_stat = _compute_V_statistic(KxzR, KyzR)

    # compute the product of the eigenvectors
    uu_prod = _compute_prod_eigvecs(KxzR, KyzR, threshold=threshold)

    if approx:
        # approximate the pvalue using the Gamma distribution
        k_appr, theta_appr = _approx_gamma_params_ci(uu_prod)
        pvalue = 1 - stats.gamma.cdf(test_stat, k_appr, 0, theta_appr)
    else:
        null_samples = _compute_null_ci(uu_prod, null_sample_size, threshold)
        pvalue = np.sum(null_samples > test_stat) / float(null_sample_size)

    return test_stat, pvalue


def _compute_V_statistic(KxR, KyR):
    # n = KxR.shape[0]
    # compute the sum of the two kernsl
    Vstat = np.sum(KxR * KyR)
    return float(Vstat)


def _approx_gamma_params_ind(Kx, Ky):
    T = Kx.shape[0]
    mean_appr = np.trace(Kx) * np.trace(Ky) / T
    var_appr = 2 * np.trace(Kx.dot(Kx)) * np.trace(Ky.dot(Ky)) / T / T
    k_appr = mean_appr**2 / var_appr
    theta_appr = var_appr / mean_appr
    return k_appr, theta_appr


def _compute_null_ind(Kx, Ky, n_samples, max_num_eigs, threshold):
    n = Kx.shape[0]

    # get the eigenvalues in ascending order, smallest to largest
    eigvals_x = np.linalg.eigvalsh(Kx)
    eigvals_y = np.linalg.eigvalsh(Ky)

    # optionally only keep the largest "N" eigenvalues
    eigvals_x = eigvals_x[-max_num_eigs:]
    eigvals_y = eigvals_y[-max_num_eigs:]
    num_eigs = len(eigvals_x)

    # compute the entry-wise product of the eigenvalues and store it as a vector
    eigvals_prod = np.dot(eigvals_x.reshape(num_eigs, 1), eigvals_y.reshape(1, num_eigs)).reshape(
        (-1, 1)
    )

    # only keep eigenvalues above a certain threshold
    eigvals_prod = eigvals_prod[eigvals_prod > eigvals_prod.max() * threshold]

    # generate chi-square distributed values z_{ij} with degree of freedom 1
    f_rand = np.random.chisquare(df=1, size=(len(eigvals_prod), n_samples))

    # compute the null distribution consisting now of (n_samples)
    # of chi-squared random variables weighted by the eigenvalue products
    null_dist = 1.0 / n * eigvals_prod.T.dot(f_rand)
    return null_dist


def _compute_prod_eigvecs(Kx, Ky, threshold):
    T = Kx.shape[0]
    wx, vx = np.linalg.eigh(0.5 * (Kx + Kx.T))
    wy, vy = np.linalg.eigh(0.5 * (Ky + Ky.T))

    if threshold is not None:
        # threshold eigenvalues that are below a certain threshold
        # and remove their corresponding values and eigenvectors
        vx = vx[:, wx > np.max(wx) * threshold]
        wx = wx[wx > np.max(wx) * threshold]
        vy = vy[:, wy > np.max(wy) * threshold]
        wy = wy[wy > np.max(wy) * threshold]

    # scale the eigenvectors by their eigenvalues
    vx = vx.dot(np.diag(np.sqrt(wx)))
    vy = vy.dot(np.diag(np.sqrt(wy)))

    # compute the product of the scaled eigenvectors
    num_eigx = vx.shape[1]
    num_eigy = vy.shape[1]
    size_u = num_eigx * num_eigy
    uu = np.zeros((T, size_u))
    for i in range(0, num_eigx):
        for j in range(0, num_eigy):
            # compute the dot product of eigenvectors
            uu[:, i * num_eigy + j] = vx[:, i] * vy[:, j]

    # now compute the product
    if size_u > T:
        uu_prod = uu.dot(uu.T)
    else:
        uu_prod = uu.T.dot(uu)

    return uu_prod


def _approx_gamma_params_ci(uu_prod):
    r"""
    Get parameters of the approximated Gamma distribution.

    Parameters
    ----------
    uu_prod : np.ndarray of shape (n_features, n_features)
        The product of the eigenvectors of Kx and Ky, the kernels
        on the input data, X and Y.

    Returns
    -------
    k_appr : float
        The shape parameter of the Gamma distribution.
    theta_appr : float
        The scale parameter of the Gamma distribution.

    Notes
    -----
    X ~ Gamma(k, theta) with a probability density function of the following:
    .. math::
        f(x; k, \\theta) = \\frac{x^{k-1} e^{-x / \\theta}}{\\theta^k \\Gamma(k)}
    where $\\Gamma(k)$ is the Gamma function evaluated at k. In this scenario
    k governs the shape of the pdf, while $\\theta$ governs more how spread out
    the data is.
    """
    # approximate the mean and the variance
    mean_appr = np.trace(uu_prod)
    var_appr = 2 * np.trace(uu_prod.dot(uu_prod))

    k_appr = mean_appr**2 / var_appr
    theta_appr = var_appr / mean_appr
    return k_appr, theta_appr


def _compute_null_ci(uu_prod, n_samples, threshold):
    # the eigenvalues of ww^T
    eig_uu = np.linalg.eigvalsh(uu_prod)
    eig_uu = eig_uu[eig_uu > eig_uu.max() * threshold]

    # generate chi-square distributed values z_{ij} with degree of freedom 1
    f_rand = np.random.chisquare(df=1, size=(eig_uu.shape[0], n_samples))

    # compute the null distribution consisting now of (n_samples)
    # of chi-squared random variables weighted by the eigenvalue products
    null_dist = eig_uu.T.dot(f_rand)
    return null_dist


def _fast_centering(k: np.ndarray) -> np.ndarray:
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


def _get_default_kernel(X: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    if X.dtype.type != np.str_:
        return partial(rbf_kernel, gamma=0.5 * estimate_squared_sigma_rbf(X))
    else:
        return delta_kernel


def _normalize_data(X: np.ndarray) -> np.ndarray:
    for column in range(X.shape[1]):
        if isinstance(X[0, column], int) or isinstance(X[0, column], float):
            X[:, column] = stats.zscore(X[:, column])
            X[:, column] = np.nan_to_num(X[:, column])  # in case some dim of X is constant

    return X
