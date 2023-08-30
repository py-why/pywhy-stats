import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.metrics.pairwise import pairwise_kernels

from pywhy_stats.kernel_utils import compute_kernel
from pywhy_stats.kernels import delta_kernel, estimate_squared_sigma_rbf


def test_given_simple_data_when_estimate_squared_sigma_rbf_then_returns_correct_results():
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator
    # 1 / (0.9 * min(std(X), IQR(X)/1.34) * n**(-1/5)) ** 2 ~= 1 / (0.9 * 2.8722813232690143* 10**(-1/5))**2 ~= 0.375890
    assert estimate_squared_sigma_rbf(X, method="silverman") == pytest.approx(0.3758902254410145)

    # 1 / median_distance_between_points**2
    assert estimate_squared_sigma_rbf(X, method="median") == pytest.approx(0.1111111111111111)


def test_delta_kernel_notworks_with_sklearn():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    Y = np.array([[1, 2, 3], [7, 8, 9], [10, 11, 12]])

    kernel = delta_kernel(X, Y)
    assert_array_equal(kernel, np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]]))

    # this should work
    kernel = compute_kernel(X, Y, kernel="delta")

    with pytest.raises(ValueError, match="Unknown kernel"):
        kernel = pairwise_kernels(X, Y, metric="delta")
        # assert_array_equal(kernel, np.array([[1, 0], [0, 0], [0, 1]]))
