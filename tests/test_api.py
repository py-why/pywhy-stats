import numpy as np
import pytest

from pywhy_stats import Methods, PValueResult, independence_test

rng = np.random.default_rng(0)


def test_independence_test():
    # Replace these arrays with appropriate test data for your specific case
    X = rng.standard_normal(size=(20, 3))  # 10 samples, 3 features for X
    Y = rng.standard_normal(size=(20, 3))  # 10 samples, 3 features for Y
    condition_on = rng.standard_normal(size=(20, 3))  # 10 samples, 3 features for condition_on

    # Test case for unconditional independence test
    result = independence_test(X, Y, method=Methods.FISHERZ)
    assert isinstance(result, PValueResult)

    # Test case for conditional independence test
    result_cond = independence_test(X, Y, condition_on=condition_on, method=Methods.KCI)
    assert isinstance(result_cond, PValueResult)

    # Add more test cases as needed for other scenarios and method combinations
    # ...

    # Test unsupported method
    with pytest.raises(ValueError, match="Invalid method type."):
        independence_test(X, Y, method="unsupported_method")

    # Test invalid data is caught with testing for Gaussianity
    n_samples = 1000
    X1 = rng.uniform(low=-5, high=0, size=(n_samples // 2, 3))  # 10 samples, 3 features for X
    X2 = rng.normal(loc=10, scale=1, size=(n_samples // 2, 3))  # 10 samples, 3 features for X
    X = np.concatenate((X1, X2), axis=0)
    Y = rng.uniform(size=(n_samples, 3))  # 10 samples, 3 features for X
    condition_on = rng.uniform(size=(n_samples, 3))  # 10 samples, 3 features for condition_on
    with pytest.warns(UserWarning, match="The provided data does not seem to be Gaussian"):
        independence_test(X, Y, condition_on, method=Methods.FISHERZ)
