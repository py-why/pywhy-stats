import numpy as np

from pywhy_stats.fisher_z_test import fisherz

seed = 12345


def test_fisher_z():
    """Test Fisher Z test for Gaussian data."""

    # We construct a SCM where X1 -> Y <- X and Y -> Z
    # so X1 is independent from X, but conditionally dependent
    # given Y or Z
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((300, 1))
    X1 = rng.standard_normal((300, 1))
    Y = X + X1 + 0.5 * rng.standard_normal((300, 1))
    Z = Y + 0.1 * rng.standard_normal((300, 1))

    # create input for the CI test
    data = np.hstack((X, X1, Y, Z))

    _, pvalue = fisherz(data, 0, 1)
    assert pvalue > 0.05
    _, pvalue = fisherz(data, 0, 1, {3})
    assert pvalue < 0.05
    _, pvalue = fisherz(data, 0, 1, {2})
    assert pvalue < 0.05
    _, pvalue = fisherz(data, 0, 3)
    assert pvalue < 0.05
    _, pvalue = fisherz(data, 0, 3, {2})
    assert pvalue > 0.05
