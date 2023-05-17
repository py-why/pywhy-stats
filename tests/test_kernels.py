import numpy as np
import pytest

from pywhy_stats.kernels import estimate_squared_sigma_rbf


def test_given_simple_data_when_estimate_squared_sigma_rbf_then_returns_correct_results():
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator
    # 1 / (0.9 * min(std(X), IQR(X)/1.34) * n**(-1/5)) ** 2 ~= 1 / (0.9 * 2.8722813232690143* 10**(-1/5))**2 ~= 0.375890
    assert estimate_squared_sigma_rbf(X, method="silverman") == pytest.approx(0.3758902254410145)

    # 1 / median_distance_between_points**2
    assert estimate_squared_sigma_rbf(X, method="median") == pytest.approx(0.1111111111111111)
