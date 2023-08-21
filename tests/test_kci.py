from functools import partial

import numpy as np
from flaky import flaky
from sklearn.metrics.pairwise import pairwise_kernels, rbf_kernel

from pywhy_stats import kci
from pywhy_stats.kernels import delta_kernel

####################################################
# Unconditional independence tests - Continuous
####################################################
seed = 12345
rng = np.random.default_rng(seed)


@flaky(max_runs=3)
def test_given_continuous_independent_data_when_perform_kernel_based_test_then_not_reject():
    n_samples = 300
    x = rng.standard_normal((n_samples, 1))
    y = np.exp(rng.uniform(size=(n_samples, 1)))

    assert kci.ind(x, y, approx=True).pvalue > 0.05
    assert kci.ind(x, y, approx=False).pvalue > 0.05


@flaky(max_runs=3)
def test_given_continuous_dependent_data_when_perform_kernel_based_test_then_reject():
    """Test a fork of variables.

    The data generating model follows the following graphical model:

         X <- Z -> Y
    """
    n_samples = 200
    z = rng.standard_normal((n_samples, 1))
    x = np.exp(z + rng.uniform(size=(n_samples, 1)))
    y = np.exp(z + rng.uniform(size=(n_samples, 1)))

    assert kci.ind(x, y, approx=True).pvalue < 0.05
    assert kci.ind(x, y, approx=False).pvalue < 0.05


####################################################
# Conditional independence tests - Continuous
####################################################


@flaky(max_runs=3)
def test_given_continuous_conditionally_independent_data_when_perform_kernel_based_test_then_not_reject():
    n_samples = 200
    z = rng.standard_normal((n_samples, 1))
    x = np.exp(z + rng.uniform(size=(n_samples, 1)))
    y = np.exp(z + rng.uniform(size=(n_samples, 1)))

    assert kci.condind(x, y, z, approx=True).pvalue > 0.05
    assert kci.condind(x, y, z, approx=False).pvalue > 0.05


@flaky(max_runs=3)
def test_given_continuous_conditionally_dependent_data_when_perform_kernel_based_test_then_reject():
    n_samples = 200
    z = rng.standard_normal((n_samples, 1))
    w = rng.standard_normal((n_samples, 1))
    x = np.exp(z + rng.uniform(size=(n_samples, 1)))
    y = np.exp(z + rng.uniform(size=(n_samples, 1)))

    assert kci.condind(x, y, w, approx=True).pvalue < 0.05
    assert kci.condind(x, y, w, approx=False).pvalue < 0.05


####################################################
# Unconditional independence tests - Categorical
####################################################


@flaky(max_runs=3)
def test_given_categorical_independent_data_when_perform_kernel_based_test_then_not_reject():
    n_samples = 200
    x = rng.normal(0, 1, n_samples)
    y = (rng.choice(2, n_samples) == 1).astype(str)

    assert kci.ind(x, y, approx=True).pvalue > 0.05
    assert kci.ind(x, y, approx=False).pvalue > 0.05


@flaky(max_runs=3)
def test_given_categorical_dependent_data_when_perform_kernel_based_test_then_reject():
    n_samples = 200
    x = rng.normal(0, 1, n_samples)
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
    n_samples = 200
    x = rng.normal(0, 1, n_samples)
    y = []

    for v in x:
        if v > 0:
            y.append(0)
        else:
            y.append(1)
    y = np.array(y).astype(str)

    w = rng.normal(0, 1, n_samples)
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
    n_samples = 200
    x = rng.normal(0, 1, n_samples)
    z = []
    for v in x:
        if v > 0:
            z.append(0)
        else:
            z.append(1)
    y = z + rng.standard_normal(len(z))
    z = np.array(z).astype(str)
    z[z == "0"] = "Class 1"
    z[z == "1"] = "Class 2"

    assert kci.condind(x, y, z, approx=True).pvalue > 0.05
    assert kci.condind(x, y, z, approx=False).pvalue > 0.05


@flaky(max_runs=3)
def test_given_categorical_conditionally_dependent_data_when_perform_kernel_based_test_then_reject():
    n_samples = 200
    x = rng.normal(0, 1, n_samples)
    z = []
    for v in x:
        if v > 0:
            z.append(0)
        else:
            z.append(1)
    y = z + rng.standard_normal(len(z))
    z = np.array(z).astype(str)
    z[z == "0"] = "Class 1"
    z[z == "1"] = "Class 2"

    assert kci.condind(x, z, y, approx=True).pvalue < 0.05
    assert kci.condind(x, z, y, approx=False).pvalue < 0.05


####################################################
# Conditional independence tests - Mixed Data Types
####################################################


@flaky(max_runs=3)
def test_given_conditionally_dependent_mixed_data_types_with_custom_kernel_when_perform_kernel_based_test_then_reject():
    n_samples = 750
    x = rng.normal(0, 1, n_samples)
    z = []
    for v in x:
        if v > 0:
            z.append(0)
        else:
            z.append(1)
    y = z + rng.standard_normal(len(z))
    z = np.array(z).astype(str)
    z[z == "0"] = "Class 1"
    z[z == "1"] = "Class 2"
    w = rng.normal(0, 1, n_samples)
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
    n_samples = 300
    X = rng.standard_normal((n_samples, 1))
    X1 = rng.standard_normal((n_samples, 1))
    Y = X + X1 + 0.5 * rng.standard_normal((n_samples, 1))
    Z = Y + 0.5 * rng.standard_normal((n_samples, 1))

    assert kci.ind(X, X1, kernel_X="linear", kernel_Y="linear").pvalue > 0.05
    assert kci.ind(X, Z, kernel_X="linear", kernel_Y="linear").pvalue < 0.05
    assert (
        kci.condind(X, Z, Y, kernel_X="linear", kernel_Y="linear", kernel_Z="linear").pvalue > 0.05
    )


@flaky(max_runs=3)
def test_given_gaussian_data_and_polynomial_kernel_when_perform_kernel_based_test_then_returns_expected_result():
    X = rng.standard_normal((300, 1))
    X1 = rng.standard_normal((300, 1))
    Y = X + X1 + 0.5 * rng.standard_normal((300, 1))
    Z = Y + 0.5 * rng.standard_normal((300, 1))

    assert kci.ind(X, X1, kernel_X="polynomial", kernel_Y="polynomial").pvalue > 0.05
    assert kci.ind(X, Z, kernel_X="polynomial", kernel_Y="polynomial").pvalue < 0.05
    assert (
        kci.condind(
            X,
            Z,
            Y,
            kernel_X="polynomial",
            kernel_Y="polynomial",
            kernel_Z="polynomial",
        ).pvalue
        > 0.05
    )

    assert (
        kci.ind(
            X, Z, kernel_X=partial(pairwise_kernels, metric="polynomial"), kernel_Y="polynomial"
        ).pvalue
        < 0.05
    )
