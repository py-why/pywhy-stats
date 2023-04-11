import flaky
import numpy as np

from pywhy_stats import fisherz


@flaky.flaky(max_runs=3, min_passes=1)
def test_fisherz_marg_ind():
    """Test FisherZ marginal independence test for Gaussian data."""
    rng = np.random.default_rng()

    # We construct a SCM where X1 -> Y <- X and Y -> Z
    # so X1 is independent from X, but conditionally dependent
    # given Y or Z
    X = rng.standard_normal((300, 1))
    X1 = rng.standard_normal((300, 1))
    Y = X + X1 + 0.5 * rng.standard_normal((300, 1))
    Z = Y + 0.1 * rng.standard_normal((300, 1))

    res = fisherz.ind(X, X1)
    assert res.pvalue > 0.05
    res = fisherz.ind(X, Z)
    assert res.pvalue < 0.05


@flaky.flaky(max_runs=3, min_passes=1)
def test_fisherz_cond_ind():
    """Test FisherZ conditional independence test for Gaussian data."""
    rng = np.random.default_rng()

    # We construct a SCM where X1 -> Y <- X and Y -> Z
    # so X1 is independent from X, but conditionally dependent
    # given Y or Z
    X = rng.standard_normal((300, 1))
    X1 = rng.standard_normal((300, 1))
    Y = X + X1 + 0.5 * rng.standard_normal((300, 1))
    Z = Y + 0.1 * rng.standard_normal((300, 1))

    res = fisherz.condind(X, X1, Z)
    assert res.pvalue < 0.05
    res = fisherz.condind(X, X1, Y)
    assert res.pvalue < 0.05
    res = fisherz.condind(X, Z, Y)
    assert res.pvalue > 0.05
