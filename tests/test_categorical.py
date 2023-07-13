from math import frexp

import numpy as np
import pandas as pd
import pytest
from dodiscover.ci import CategoricalCITest, GSquareCITest
from dodiscover.testdata import testdata
from numpy.testing import assert_almost_equal

seed = 12345

df_adult = pd.read_csv("tests/testdata/adult.csv")


def test_chisquare_marginal_independence_adult_dataset():
    """Test that chi-square tests return the correct answer for marginal independence queries.

    Uses the test data from dagitty.
    """
    # Comparision values taken from dagitty (DAGitty)
    ci_est = CategoricalCITest("pearson")
    coef, p_value = ci_est.test(x_vars={"Age"}, y_vars={"Immigrant"}, z_covariates=[], df=df_adult)
    assert_almost_equal(coef, 57.75, decimal=1)
    assert_almost_equal(np.log(p_value), -25.47, decimal=1)
    assert ci_est.dof_ == 4

    coef, p_value = ci_est.test(x_vars={"Age"}, y_vars={"Race"}, z_covariates=[], df=df_adult)
    assert_almost_equal(coef, 56.25, decimal=1)
    assert_almost_equal(np.log(p_value), -24.75, decimal=1)
    assert ci_est.dof_ == 4

    coef, p_value = ci_est.test(x_vars={"Age"}, y_vars={"Sex"}, z_covariates=[], df=df_adult)
    assert_almost_equal(coef, 289.62, decimal=1)
    assert_almost_equal(np.log(p_value), -139.82, decimal=1)
    assert ci_est.dof_ == 4

    coef, p_value = ci_est.test(x_vars={"Immigrant"}, y_vars={"Sex"}, z_covariates={}, df=df_adult)
    assert_almost_equal(coef, 0.2724, decimal=1)
    assert_almost_equal(np.log(p_value), -0.50, decimal=1)
    assert ci_est.dof_ == 1


def test_chisquare_conditional_independence_adult_dataset():
    """Test that chi-square tests return the correct answer for conditional independence queries.

    Uses the test data from dagitty.
    """
    ci_est = CategoricalCITest("pearson")

    coef, p_value = coef, p_value = ci_est.test(
        x_vars={"Education"},
        y_vars={"HoursPerWeek"},
        z_covariates=["Age", "Immigrant", "Race", "Sex"],
        df=df_adult,
    )
    assert_almost_equal(coef, 1460.11, decimal=1)
    assert_almost_equal(p_value, 0, decimal=1)
    assert ci_est.dof_ == 316

    coef, p_value = ci_est.test(
        x_vars={"Education"}, y_vars={"MaritalStatus"}, z_covariates=["Age", "Sex"], df=df_adult
    )
    assert_almost_equal(coef, 481.96, decimal=1)
    assert_almost_equal(p_value, 0, decimal=1)
    assert ci_est.dof_ == 58

    # Values differ (for next 2 tests) from dagitty because dagitty ignores grouped
    # dataframes with very few samples. Update: Might be same from scipy_vars=1.7.0
    coef, p_value = ci_est.test(
        x_vars={"Income"},
        y_vars={"Race"},
        z_covariates=["Age", "Education", "HoursPerWeek", "MaritalStatus"],
        df=df_adult,
    )

    assert_almost_equal(coef, 66.39, decimal=1)
    assert_almost_equal(p_value, 0.99, decimal=1)
    assert ci_est.dof_ == 136

    coef, p_value = ci_est.test(
        x_vars={"Immigrant"},
        y_vars={"Income"},
        z_covariates=["Age", "Education", "HoursPerWeek", "MaritalStatus"],
        df=df_adult,
    )
    assert_almost_equal(coef, 65.59, decimal=1)
    assert_almost_equal(p_value, 0.999, decimal=2)
    assert ci_est.dof_ == 131


@pytest.mark.parametrize(
    "ci_test",
    [
        CategoricalCITest("pearson"),  # chi-square
        CategoricalCITest("log-likelihood"),  # G^2
        CategoricalCITest("freeman-tukey"),  # freeman-tukey
        CategoricalCITest("mod-log-likelihood"),  # Modified log-likelihood
        CategoricalCITest("neyman"),  # Neyman
        CategoricalCITest("cressie-read"),  # Cressie-read
    ],
)
def test_chisquare_when_dependent(ci_test):
    assert (
        ci_test.test(
            x_vars={"Age"},
            y_vars={"Immigrant"},
            z_covariates=[],
            df=df_adult,
        )[1]
        < 0.05
    )

    assert (
        ci_test.test(
            x_vars={"Age"},
            y_vars={"Race"},
            z_covariates=[],
            df=df_adult,
        )[1]
        < 0.05
    )

    assert (
        ci_test.test(
            x_vars={"Age"},
            y_vars={"Sex"},
            z_covariates=[],
            df=df_adult,
        )[1]
        < 0.05
    )
    assert (
        ci_test.test(
            x_vars={"Immigrant"},
            y_vars={"Sex"},
            z_covariates=[],
            df=df_adult,
        )[1]
        >= 0.05
    )

    assert (
        ci_test.test(
            x_vars={"Education"},
            y_vars={"HoursPerWeek"},
            z_covariates=["Age", "Immigrant", "Race", "Sex"],
            df=df_adult,
        )[1]
        < 0.05
    )
    assert (
        ci_test.test(
            x_vars={"Education"},
            y_vars={"MaritalStatus"},
            z_covariates=["Age", "Sex"],
            df=df_adult,
        )[1]
        < 0.05
    )


@pytest.mark.parametrize(
    "ci_test",
    [
        CategoricalCITest("pearson"),  # chi-square
        CategoricalCITest("log-likelihood"),  # G^2
        CategoricalCITest("freeman-tukey"),  # freeman-tukey
        CategoricalCITest("mod-log-likelihood"),  # Modified log-likelihood
        CategoricalCITest("neyman"),  # Neyman
        CategoricalCITest("cressie-read"),  # Cressie-read
    ],
)
def test_chisquare_when_exactly_dependent(ci_test):
    x = np.random.choice([0, 1], size=1000)
    y = x.copy()
    df = pd.DataFrame({"x": x, "y": y})

    stat, p_value = ci_test.test(x_vars={"x"}, y_vars={"y"}, z_covariates=[], df=df)
    assert ci_test.dof_ == 1
    assert_almost_equal(p_value, 0, decimal=5)


def test_g_discrete():
    """Test G^2 test for discrete data."""
    dm = np.array([testdata.dis_data]).reshape((10000, 5))
    x = 0
    y = 1
    ci_estimator = GSquareCITest(data_type="discrete", levels=[3, 2, 3, 4, 2])
    df = pd.DataFrame.from_records(dm)

    sets = [[], [2], [2, 3], [3, 4], [2, 3, 4]]
    for idx in range(len(sets)):
        _, p = ci_estimator.test(df, {x}, {y}, set(sets[idx]))
        fr_p = frexp(p)
        fr_a = frexp(testdata.dis_answer[idx])

        # due to adding small perturbation to prevent dividing by 0 within
        # G^2 statistic computation
        assert round(fr_p[0] - fr_a[0], 3) == 0
        assert fr_p[1] == fr_a[1]
        assert fr_p[0] > 0

    # check error message for number of samples
    dm = np.array([testdata.dis_data]).reshape((2000, 25))
    df = pd.DataFrame.from_records(dm)
    levels = np.ones((25,)) * 3
    ci_estimator = GSquareCITest(data_type="discrete", levels=levels)
    sets = [[2, 3, 4, 5, 6, 7]]
    with pytest.raises(RuntimeError, match="Not enough samples"):
        ci_estimator.test(df, {x}, {y}, set(sets[0]))


def test_g_binary():
    """Test G^2 test for binary data."""
    dm = np.array([testdata.bin_data]).reshape((5000, 5))
    x = 0
    y = 1
    ci_estimator = GSquareCITest(data_type="binary")
    df = pd.DataFrame.from_records(dm)

    sets = [[], [2], [2, 3], [3, 4], [2, 3, 4]]
    for idx in range(len(sets)):
        _, p = ci_estimator.test(df, {x}, {y}, set(sets[idx]))
        fr_p = frexp(p)
        fr_a = frexp(testdata.bin_answer[idx])
        assert fr_p[1] == fr_a[1]
        assert round(fr_p[0] - fr_a[0], 4) == 0
        assert fr_p[0] > 0

    # check error message for number of samples
    dm = np.array([testdata.bin_data]).reshape((500, 50))
    df = pd.DataFrame.from_records(dm)
    sets = [[2, 3, 4, 5, 6, 7, 8]]
    with pytest.raises(RuntimeError, match="Not enough samples"):
        ci_estimator.test(df, {x}, {y}, set(sets[0]))


def binary_scm(n_samples=200):
    # We construct a SCM where X1 -> Y <- X and Y -> Z
    # so X1 is independent from X, but conditionally dependent
    # given Y or Z
    rng = np.random.default_rng(seed)

    X = rng.binomial(1, 0.3, (n_samples, 1))
    X1 = rng.binomial(1, 0.6, (n_samples, 1))
    Y = X * X1
    Z = Y + (1 - rng.binomial(1, 0.5, (n_samples, 1)))

    # create input for the CI test
    df = pd.DataFrame(np.hstack((X, X1, Y, Z)), columns=["x", "x1", "y", "z"])
    return df


def test_g_binary_simulation():
    """Test G^2 test for binary data."""
    rng = np.random.default_rng(seed)
    n_samples = 500
    df = binary_scm(n_samples=n_samples)
    for i in range(10):
        df[i] = rng.binomial(1, p=0.5, size=n_samples)
    ci_estimator = GSquareCITest(data_type="binary")

    _, pvalue = ci_estimator.test(df, {"x"}, {"y"})
    assert pvalue < 0.05
    _, pvalue = ci_estimator.test(df, {"x1"}, {"y"})
    assert pvalue < 0.05
    _, pvalue = ci_estimator.test(df, {"x"}, {"x1"})
    assert pvalue > 0.05
    _, pvalue = ci_estimator.test(df, {"x1"}, {0})
    assert pvalue > 0.05

    _, pvalue = ci_estimator.test(df, {"x"}, {"x1"}, {"y"})
    assert pvalue < 0.05


def test_g_binary_highdim():
    """Test G^2 test for binary data."""
    rng = np.random.default_rng(seed)
    n_samples = 1000
    df = binary_scm(n_samples=n_samples)
    for i in range(10):
        df[i] = rng.binomial(1, p=0.8, size=n_samples)
    ci_estimator = GSquareCITest(data_type="binary")

    _, pvalue = ci_estimator.test(df, {"x"}, {"x1"}, set(range(6)))
    assert pvalue > 0.05
    _, pvalue = ci_estimator.test(df, {"x"}, {"y"}, set(range(5)).union({"x1"}))
    assert pvalue < 0.05
