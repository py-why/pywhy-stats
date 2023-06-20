import numpy as np
from flaky import flaky
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel

from pywhy_stats import kci
from pywhy_stats.kernels import delta_kernel

####################################################
# Unconditional independence tests - Continuous
####################################################


@flaky(max_runs=3)
def test_given_continuous_independent_data_when_perform_kernel_based_test_then_not_reject():
    x = np.random.randn(1000, 1)
    y = np.exp(np.random.rand(1000, 1))

    assert kci.ind(x, y, approx=True).pvalue > 0.05
    assert kci.ind(x, y, approx=False).pvalue > 0.05


@flaky(max_runs=3)
def test_given_continuous_dependent_data_when_perform_kernel_based_test_then_reject():
    """Test a fork of variables.

    The data generating model follows the following graphical model:

         X <- Z -> Y
    """
    z = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert kci.ind(x, y, approx=True).pvalue < 0.05
    assert kci.ind(x, y, approx=False).pvalue < 0.05


####################################################
# Conditional independence tests - Continuous
####################################################


@flaky(max_runs=3)
def test_given_continuous_conditionally_independent_data_when_perform_kernel_based_test_then_not_reject():
    z = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert kci.condind(x, y, z, approx=True).pvalue > 0.05
    assert kci.condind(x, y, z, approx=False).pvalue > 0.05


@flaky(max_runs=3)
def test_given_continuous_conditionally_dependent_data_when_perform_kernel_based_test_then_reject():
    z = np.random.randn(1000, 1)
    w = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert kci.condind(x, y, w, approx=True).pvalue < 0.05
    assert kci.condind(x, y, w, approx=False).pvalue < 0.05


####################################################
# Unconditional independence tests - Categorical
####################################################


@flaky(max_runs=3)
def test_given_categorical_independent_data_when_perform_kernel_based_test_then_not_reject():
    x = np.random.normal(0, 1, 1000)
    y = (np.random.choice(2, 1000) == 1).astype(str)

    assert kci.ind(x, y, approx=True).pvalue > 0.05
    assert kci.ind(x, y, approx=False).pvalue > 0.05


@flaky(max_runs=3)
def test_given_categorical_dependent_data_when_perform_kernel_based_test_then_reject():
    x = np.random.normal(0, 1, 1000)
    y = []

    for v in x:
        if v > 0:
            y.append(0)
        else:
            y.append(1)
    y = np.array(y).astype(str)

    assert kci.ind(x, y, approx=True).pvalue < 0.05
    assert kci.ind(x, y, approx=False).pvalue < 0.05


@flaky(max_runs=3)
def test_given_dependent_mixed_data_types_when_perform_kernel_based_test_then_reject():
    x = np.random.normal(0, 1, 1000)
    y = []

    for v in x:
        if v > 0:
            y.append(0)
        else:
            y.append(1)
    y = np.array(y).astype(str)

    w = np.random.normal(0, 1, 1000)
    y = np.vstack([y, w]).T

    def my_custom_kernel(X):
        # X here is multidimensional, where the first dimension is continuous and the second categorical. We can handle
        # this by combining two kernels.
        return delta_kernel(X[:, 0].reshape(-1, 1)) * rbf_kernel(X[:, 1].reshape(-1, 1))

    assert kci.ind(x, y, kernel_Y=my_custom_kernel, approx=True).pvalue < 0.05
    assert kci.ind(x, y, kernel_Y=my_custom_kernel, approx=False).pvalue < 0.05


####################################################
# Conditional independence tests - Categorical
####################################################


@flaky(max_runs=3)
def test_given_categorical_conditionally_independent_data_when_perform_kernel_based_test_then_not_reject():
    x = np.random.normal(0, 1, 1000)
    z = []
    for v in x:
        if v > 0:
            z.append(0)
        else:
            z.append(1)
    y = z + np.random.randn(len(z))
    z = np.array(z).astype(str)
    z[z == "0"] = "Class 1"
    z[z == "1"] = "Class 2"

    assert kci.condind(x, y, z, approx=True).pvalue > 0.05
    assert kci.condind(x, y, z, approx=False).pvalue > 0.05


@flaky(max_runs=3)
def test_given_categorical_conditionally_dependent_data_when_perform_kernel_based_test_then_reject():
    x = np.random.normal(0, 1, 1000)
    z = []
    for v in x:
        if v > 0:
            z.append(0)
        else:
            z.append(1)
    y = z + np.random.randn(len(z))
    z = np.array(z).astype(str)
    z[z == "0"] = "Class 1"
    z[z == "1"] = "Class 2"

    assert kci.condind(x, z, y, approx=True).pvalue < 0.05
    assert kci.condind(x, z, y, approx=False).pvalue < 0.05


@flaky(max_runs=3)
def test_given_conditionally_dependent_mixed_data_types_with_custom_kernel_when_perform_kernel_based_test_then_reject():
    x = np.random.normal(0, 1, 1000)
    z = []
    for v in x:
        if v > 0:
            z.append(0)
        else:
            z.append(1)
    y = z + np.random.randn(len(z))
    z = np.array(z).astype(str)
    z[z == "0"] = "Class 1"
    z[z == "1"] = "Class 2"
    w = np.random.normal(0, 1, 1000)
    z = np.vstack([z, w]).T

    def my_custom_kernel(X):
        # X here is multidimensional, where the first dimension is continuous and the second categorical. We can handle
        # this by combining two kernels.
        return delta_kernel(X[:, 0].reshape(-1, 1)) * rbf_kernel(X[:, 1].reshape(-1, 1))

    assert kci.condind(x, z, y, kernel_Y=my_custom_kernel, approx=True).pvalue < 0.05
    assert kci.condind(x, z, y, kernel_Y=my_custom_kernel, approx=False).pvalue < 0.05


####################################################
# Testing different kernels
####################################################


@flaky(max_runs=3)
def test_given_gaussian_data_and_linear_kernel_when_perform_kernel_based_test_then_returns_expected_result():
    X = np.random.randn(300, 1)
    X1 = np.random.randn(300, 1)
    Y = X + X1 + 0.5 * np.random.randn(300, 1)
    Z = Y + 0.5 * np.random.randn(300, 1)

    assert kci.ind(X, X1, kernel_X=linear_kernel, kernel_Y=linear_kernel).pvalue > 0.05
    assert kci.ind(X, Z, kernel_X=linear_kernel, kernel_Y=linear_kernel).pvalue < 0.05
    assert (
        kci.condind(
            X, Z, Y, kernel_X=linear_kernel, kernel_Y=linear_kernel, kernel_Z=linear_kernel
        ).pvalue
        > 0.05
    )


@flaky(max_runs=3)
def test_given_gaussian_data_and_polynomial_kernel_when_perform_kernel_based_test_then_returns_expected_result():
    X = np.random.randn(300, 1)
    X1 = np.random.randn(300, 1)
    Y = X + X1 + 0.5 * np.random.randn(300, 1)
    Z = Y + 0.5 * np.random.randn(300, 1)

    assert kci.ind(X, X1, kernel_X=polynomial_kernel, kernel_Y=polynomial_kernel).pvalue > 0.05
    assert kci.ind(X, Z, kernel_X=polynomial_kernel, kernel_Y=polynomial_kernel).pvalue < 0.05
    assert (
        kci.condind(
            X,
            Z,
            Y,
            kernel_X=polynomial_kernel,
            kernel_Y=polynomial_kernel,
            kernel_Z=polynomial_kernel,
        ).pvalue
        > 0.05
    )
