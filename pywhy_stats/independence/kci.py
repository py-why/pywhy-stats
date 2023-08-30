"""Independence test using Kernel test.

Examples
--------
>>> import pywhy_stats as ps
>>> res = ps.kci.ind([1, 2, 3], [4, 5, 6])
>>> print(res.pvalue)
>>> 1.0
"""
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from pywhy_stats.kernel_utils import _fast_centering, _preprocess_kernel_data, compute_kernel
from pywhy_stats.utils import preserve_random_state

from ..pvalue_result import PValueResult


@preserve_random_state
def ind(
    X: np.ndarray,
    Y: np.ndarray,
    kernel_X: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    kernel_Y: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    approx: bool = True,
    null_sample_size: int = 1000,
    threshold: float = 1e-5,
    centered: bool = True,
    normalize_data: bool = True,
    n_jobs: Optional[int] = None,
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
    kernel_X : Callable[[ArrayLike], ArrayLike], str
        The kernel function for X. By default, the RBF kernel is used for numeric and the delta
        kernel for categorical data. Note that we currently only consider string values as categorical data.
        For more information on how to set the kernel, see the Notes.
    kernel_Y : Callable[[ArrayLike], ArrayLike], str
        The kernel function for Y. By default, the RBF kernel is used for continuous and the delta
        kernel for categorical data. Note that we currently only consider string values as categorical data.
        For more information on how to set the kernel, see the Notes.
    approx : bool
        Whether to use the Gamma distribution approximation for the pvalue, by default True.
    null_sample_size : int
        The number of samples to generate for the bootstrap distribution to approximate the pvalue,
        by default 1000.
    threshold : float
        The threshold set on the value of eigenvalues, by default 1e-5. Used to regularize the
        method.
    centered : bool
        Whether to center the kernel matrix or not, by default True.
    normalize_data : bool
        Whether the data should be standardized to unit variance, by default True.
    n_jobs : Optional[int], optional
        The number of jobs to run computations in parallel, by default None.
    random_seed : Optional[int], optional
        Random seed, by default None.

    Notes
    -----
    Any callable can be given to create the kernel matrix. For instance, to use a particular kernel
    from sklearn::

        kernel_X = func:`sklearn.metrics.pairwise.pairwise_kernels.polynomial`

    If the kernel is a callable, it will have either one input for ``X``, or two inputs for ``X`` and
    ``Y``. It is assumed that the kernel operates on the entire array to compute
    the kernel array, that is, the callable performs a vectorized operation to compute the entire kernel matrix.

    If one has an unvectorizable kernel function, then it is advised to use the callable with the
    :func:`~sklearn.metrics.pairwise.pairwise_kernels` function, which will parallelize the kernel computation
    across each row of the data.

    References
    ----------
    .. footbibliography::
    """
    test_statistic, pvalue = _kernel_test(
        X,
        Y,
        None,
        kernel_X,
        kernel_Y,
        None,
        approx,
        null_sample_size,
        threshold,
        centered=centered,
        normalize_data=normalize_data,
        n_jobs=n_jobs,
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
    centered: bool = True,
    normalize_data: bool = True,
    n_jobs: Optional[int] = None,
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
        For more information on how to set the kernel, see the Notes.
    kernel_Y : Callable[[ArrayLike], ArrayLike]
        The kernel function for Y. By default, the RBF kernel is used for continuous and the delta
        kernel for categorical data. Note that we currently only consider string values as categorical data.
        For more information on how to set the kernel, see the Notes.
    kernel_Z : Callable[[ArrayLike], ArrayLike]
        The kernel function for Z. By default, the RBF kernel is used for continuous and the delta
        kernel for categorical data. Note that we currently only consider string values as categorical data.
        For more information on how to set the kernel, see the Notes.
    approx : bool
        Whether to use the Gamma distribution approximation for the pvalue, by default True.
    null_sample_size : int
        The number of samples to generate for the bootstrap distribution to approximate the pvalue,
        by default 1000.
    threshold : float
        The threshold set on the value of eigenvalues, by default 1e-5. Used to regularize the
        method.
    centered : bool
        Whether to center the kernel matrices, by default True.
    normalize_data : bool
        Whether the data should be standardized to unit variance, by default True.
    n_jobs : Optional[int], optional
        The number of jobs to run computations in parallel, by default None.
    random_seed : Optional[int], optional
        Random seed, by default None.

    Notes
    -----
    Any callable can be given to create the kernel matrix. For instance, to use a particular kernel
    from sklearn::

        kernel_X = func:`sklearn.metrics.pairwise.pairwise_kernels.polynomial`

    In addition, we implement an efficient delta kernel. The delta kernel can be specified using the
    'kernel' string argument.

    If the kernel is a callable, it will have either one input for ``X``, or two inputs for ``X`` and
    ``Y``. It is assumed that the kernel operates on the entire array to compute
    the kernel array, that is, the callable performs a vectorized operation to compute the entire kernel matrix.

    If one has an unvectorizable kernel function, then it is advised to use the callable with the
    :func:`~sklearn.metrics.pairwise.pairwise_kernels` function, which will parallelize the kernel computation
    across each row of the data.

    References
    ----------
    .. footbibliography::
    """
    test_statistic, pvalue = _kernel_test(
        X,
        Y,
        Z,
        kernel_X,
        kernel_Y,
        kernel_Z,
        approx,
        null_sample_size,
        threshold,
        centered=centered,
        normalize_data=normalize_data,
        n_jobs=n_jobs,
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
    centered: bool,
    n_jobs: Optional[int],
) -> Tuple[float, float]:
    X, Y, Z = _preprocess_kernel_data(X, Y, Z, normalize_data)

    Ky = compute_kernel(Y, kernel=kernel_Y, centered=centered, n_jobs=n_jobs)

    # compute kernels in each data space
    if Z is not None:
        Kz = compute_kernel(Z, kernel=kernel_Z, centered=False, n_jobs=n_jobs)

        Kx = compute_kernel(X, kernel=kernel_X, centered=False, n_jobs=n_jobs)
        Kx *= Kz  # type: ignore

        if centered:
            Kx = _fast_centering(Kx)
            Kz = _fast_centering(Kz)

        return _cond(Kx, Ky, Kz, approx, null_sample_size, threshold)
    else:
        Kx = compute_kernel(X, kernel=kernel_X, centered=centered, n_jobs=n_jobs)

        return _ind(
            Kx,
            Ky,
            approx,
            null_sample_size,
            threshold,
        )


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
