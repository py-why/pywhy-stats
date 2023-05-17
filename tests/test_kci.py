import numpy as np
from flaky import flaky

from pywhy_stats import kci
from pywhy_stats.custom_kernels import delta_kernel


@flaky(max_runs=3)
def test_given_continuous_independent_data_when_perform_kernel_based_test_then_not_reject():
    x = np.random.randn(1000, 1)
    y = np.exp(np.random.rand(1000, 1))

    assert kci.ind(x, y).pvalue > 0.05


@flaky(max_runs=3)
def test_given_continuous_dependent_data_when_perform_kernel_based_test_then_reject():
    z = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert kci.ind(x, y).pvalue < 0.05


@flaky(max_runs=3)
def test_given_categorical_independent_data_when_perform_kernel_based_test_then_not_reject():
    x = np.random.normal(0, 1, 1000)
    y = (np.random.choice(2, 1000) == 1).astype(str)

    assert kci.ind(x, y, kernel_Y=delta_kernel).pvalue > 0.05


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

    assert kci.ind(x, y, kernel_Y=delta_kernel).pvalue < 0.05
