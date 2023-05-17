from typing import Callable

import numpy as np
from numpy import shape, ndarray
from numpy.linalg import eigvalsh
from numpy.random import permutation
from scipy import stats

from pywhy_stats.custom_kernels import rbf_kernel
from pywhy_stats.pvalue_result import PValueResult


def ind(X: np.ndarray,
        Y: np.ndarray,
        kernel_X: Callable[[np.ndarray], np.ndarray] = rbf_kernel,
        kernel_Y: Callable[[np.ndarray], np.ndarray] = rbf_kernel,
        approx: bool = True,
        null_ss: int = 1000) -> PValueResult:
    """
    Python implementation of Kernel-based Conditional Independence (KCI) test and the unconditional version.
    The original Matlab implementation can be found in http://people.tuebingen.mpg.de/kzhang/KCI-test.zip

    References
    ----------
    [1] K. Zhang, J. Peters, D. Janzing, and B. Schölkopf,
    "A kernel-based conditional independence test and application in causal discovery," In UAI 2011.
    [2] A. Gretton, K. Fukumizu, C.-H. Teo, L. Song, B. Schölkopf, and A. Smola, "A kernel
    Statistical test of independence." In NIPS 21, 2007.

    Parameters
    ----------
    X: numpy array of shape (n_samples, n_columns)
    Y: numpy array of shape (n_samples, n_columns)
    kernel_X: callable expecting nxd1 array returning nxn array
              callable function that converts X into a kernel matrix
    kernel_Y: callable expecting nxd1 array returning nxn array
              callable function that converts Y into a kernel matrix
    approx: boolean (default: True)
            whether to use gamma approximation
    null_ss: int (default: 1000)
             sample size in simulating the null distribution

    Returns
    -------
    pvalue_result: PValueResult object
                   Contains p-value and the corresponding test statistic.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    Kx, Ky = _ind_kernel_matrix(X, Y, kernel_X, kernel_Y)
    test_stat, Kxc, Kyc = _hsic_v_statistic(Kx, Ky)

    if approx:
        k_appr, theta_appr = _get_kappa(Kxc, Kyc)
        pvalue = 1 - stats.gamma.cdf(test_stat, k_appr, 0, theta_appr)
    else:
        null_dstr = _null_sample_spectral(Kxc, Kyc, null_ss)
        pvalue = sum(null_dstr.squeeze() > test_stat) / float(null_ss)

    return PValueResult(pvalue=pvalue, statistic=test_stat)


def condind(X, Y, Z):
    raise NotImplementedError("Not yet implemented")


def _ind_kernel_matrix(X: np.ndarray,
                       Y: np.ndarray,
                       kernel_X: Callable[[np.ndarray], np.ndarray],
                       kernel_Y: Callable[[np.ndarray], np.ndarray]):
    """
    Compute kernel matrix for data x and data y

    Parameters
    ----------
    X: input data for x (nxd1 array)
    Y: input data for y (nxd1 array)
    kernel_X: callable function that converts X into a kernel matrix (Callable expecting nxd1 array returning nxn array)
    kernel_Y: callable function that converts Y into a kernel matrix (Callable expecting nxd1 array returning nxn array)

    Returns
    Kx: kernel matrix for data_x (nxn)
    Ky: kernel matrix for data_y (nxn)
    """
    if X.dtype.type != np.str_:
        X = stats.zscore(X, ddof=1, axis=0)
        X[np.isnan(X)] = 0.  # in case some dim of data_x is constant
    if Y.dtype.type != np.str_:
        Y = stats.zscore(Y, ddof=1, axis=0)
        Y[np.isnan(Y)] = 0.  # in case some dim of data_x is constant
    # We set 'ddof=1' to conform to the normalization way in the original Matlab implementation in
    # http://people.tuebingen.mpg.de/kzhang/KCI-test.zip

    return kernel_X(X), kernel_Y(Y)


def _hsic_v_statistic(Kx, Ky):
    """
    Compute V test statistic from kernel matrices Kx and Ky
    Parameters
    ----------
    Kx: kernel matrix for data_x (nxn)
    Ky: kernel matrix for data_y (nxn)

    Returns
    _________
    Vstat: HSIC v statistics
    Kxc: centralized kernel matrix for data_x (nxn)
    Kyc: centralized kernel matrix for data_y (nxn)
    """
    Kxc = _center_kernel_matrix(Kx)
    Kyc = _center_kernel_matrix(Ky)
    return np.sum(Kxc * Kyc), Kxc, Kyc


def _center_kernel_matrix(K: ndarray):
    """
    Centers the kernel matrix via a centering matrix H=I-1/n and returns HKH
    [Updated @Haoyue 06/24/2022]
    equivalent to:
        H = eye(n) - 1.0 / n
        return H.dot(K.dot(H))
    since n is always big, we can save time on the dot product by plugging H into dot and expand as sum.
    time complexity is reduced from O(n^3) (matrix dot) to O(n^2) (traverse each element).
    Also, consider the fact that here K (both Kx and Ky) are symmetric matrices, so K_colsums == K_rowsums
    """
    # assert np.all(K == K.T), 'K should be symmetric'
    n = shape(K)[0]
    K_colsums = K.sum(axis=0)
    K_allsum = K_colsums.sum()
    return K - (K_colsums[None, :] + K_colsums[:, None]) / n + (K_allsum / n ** 2)


def _get_kappa(Kx, Ky):
    """
    Get parameters for the approximated gamma distribution
    Parameters
    ----------
    Kx: kernel matrix for data_x (nxn)
    Ky: kernel matrix for data_y (nxn)

    Returns
    _________
    k_appr, theta_appr: approximated parameters of the gamma distribution

    [Updated @Haoyue 06/24/2022]
    equivalent to:
        var_appr = 2 * np.trace(Kx.dot(Kx)) * np.trace(Ky.dot(Ky)) / T / T
    based on the fact that:
        np.trace(K.dot(K)) == np.sum(K * K.T), where here K is symmetric
    we can save time on the dot product by only considering the diagonal entries of K.dot(K)
    time complexity is reduced from O(n^3) (matrix dot) to O(n^2) (traverse each element),
    where n is usually big (sample size).
    """
    T = Kx.shape[0]
    mean_appr = np.trace(Kx) * np.trace(Ky) / T
    var_appr = 2 * np.sum(Kx ** 2) * np.sum(Ky ** 2) / T / T  # same as np.sum(Kx * Kx.T) ..., here Kx is symmetric
    k_appr = mean_appr ** 2 / var_appr
    theta_appr = var_appr / mean_appr
    return k_appr, theta_appr


def _null_sample_spectral(Kxc, Kyc, null_ss, threshold=1e-6):
    """
    Simulate data from null distribution

    Parameters
    ----------
    Kxc: centralized kernel matrix for data_x (nxn)
    Kyc: centralized kernel matrix for data_y (nxn)

    Returns
    _________
    null_dstr: samples from the null distribution

    """
    T = Kxc.shape[0]
    if T > 1000:
        num_eig = np.int(np.floor(T / 2))
    else:
        num_eig = T
    lambdax = eigvalsh(Kxc)
    lambday = eigvalsh(Kyc)
    lambdax = -np.sort(-lambdax)
    lambday = -np.sort(-lambday)
    lambdax = lambdax[0:num_eig]
    lambday = lambday[0:num_eig]
    lambda_prod = np.dot(lambdax.reshape(num_eig, 1), lambday.reshape(1, num_eig)).reshape(
        (num_eig ** 2, 1))
    lambda_prod = lambda_prod[lambda_prod > lambda_prod.max() * threshold]
    f_rand = np.random.chisquare(1, (lambda_prod.shape[0], null_ss))
    null_dstr = lambda_prod.T.dot(f_rand) / T
    return null_dstr
