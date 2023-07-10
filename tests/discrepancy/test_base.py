import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pywhy_stats.discrepancy.base import compute_null


# Define a dummy test statistic function for testing
def _test_statistic(X, Y, group_ind, **kwargs):
    # Return the sum of X and Y as the test statistic
    return np.sum(X[group_ind]) + np.sum(Y[group_ind])


def test_compute_null():
    # Set up test data
    e_hat = np.array([0.7, 0.5, 0.3])
    X = np.array([[1, 2], [3, 4], [5, 6]])
    Y = np.array([[7, 8], [9, 10], [11, 12]])

    rng = np.random.default_rng()

    # Call the compute_null function
    null_dist = compute_null(_test_statistic, e_hat, X, Y, null_reps=10, n_jobs=-1, seed=rng)

    # Check the shape of the null_dist array
    assert null_dist.shape == (10,)

    # The null distribution should correctly sample some variation
    assert len(np.unique(null_dist)) > 1
