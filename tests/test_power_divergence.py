from math import frexp

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from pywhy_stats import power_divergence

from .testdata import testdata

seed = 12345

df_adult = pd.read_csv("tests/testdata/adult.csv")


def _binary_scm(n_samples=200):
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


def test_chisquare_marginal_independence_adult_dataset():
    """Test that chi-square tests return the correct answer for marginal independence queries.

    Uses the test data from dagitty.
    """
    # Comparision values taken from dagitty (DAGitty)
    method = "pearson"
    X = df_adult["Age"]
    Y = df_adult["Immigrant"]
    result = power_divergence.ind(X=X, Y=Y, method=method)
    assert_almost_equal(result.statistic, 57.75, decimal=1)
    assert_almost_equal(np.log(result.pvalue), -25.47, decimal=1)
    assert result.additional_information["dof"] == 4

    Y = df_adult["Race"]
    result = power_divergence.ind(X=X, Y=Y, method=method)
    assert_almost_equal(result.statistic, 56.25, decimal=1)
    assert_almost_equal(np.log(result.pvalue), -24.75, decimal=1)
    assert result.additional_information["dof"] == 4

    Y = df_adult["Sex"]
    result = power_divergence.ind(X=X, Y=Y, method=method)
    assert_almost_equal(result.statistic, 289.62, decimal=1)
    assert_almost_equal(np.log(result.pvalue), -139.82, decimal=1)
    assert result.additional_information["dof"] == 4

    X = df_adult["Immigrant"]
    Y = df_adult["Sex"]
    result = power_divergence.ind(X=X, Y=Y, method=method)
    assert_almost_equal(result.statistic, 0.2724, decimal=1)
    assert_almost_equal(np.log(result.pvalue), -0.50, decimal=1)
    assert result.additional_information["dof"] == 1


def test_chisquare_conditional_independence_adult_dataset():
    """Test that chi-square tests return the correct answer for conditional independence queries.

    Uses the test data from dagitty.
    """
    method = "pearson"
    X = df_adult["Education"]
    Y = df_adult["HoursPerWeek"]
    condition_on = df_adult[["Age", "Immigrant", "Race", "Sex"]]
    result = power_divergence.condind(X=X, Y=Y, condition_on=condition_on, method=method)
    assert_almost_equal(result.statistic, 1460.11, decimal=1)
    assert_almost_equal(result.pvalue, 0, decimal=1)
    assert result.additional_information["dof"] == 316

    Y = df_adult["MaritalStatus"]
    condition_on = df_adult[["Age", "Sex"]]
    result = power_divergence.condind(X=X, Y=Y, condition_on=condition_on, method=method)
    assert_almost_equal(result.statistic, 481.96, decimal=1)
    assert_almost_equal(result.pvalue, 0, decimal=1)
    assert result.additional_information["dof"] == 58

    # Values differ (for next 2 tests) from dagitty because dagitty ignores grouped
    # dataframes with very few samples. Update: Might be same from scipy_vars=1.7.0
    X = df_adult["Income"]
    Y = df_adult["Race"]
    condition_on = df_adult[["Age", "Education", "HoursPerWeek", "MaritalStatus"]]
    result = power_divergence.condind(X=X, Y=Y, condition_on=condition_on, method=method)

    assert_almost_equal(result.statistic, 66.39, decimal=1)
    assert_almost_equal(result.pvalue, 0.99, decimal=1)
    assert result.additional_information["dof"] == 136

    X = df_adult["Immigrant"]
    Y = df_adult["Income"]
    condition_on = df_adult[["Age", "Education", "HoursPerWeek", "MaritalStatus"]]
    result = power_divergence.condind(X=X, Y=Y, condition_on=condition_on, method=method)
    assert_almost_equal(result.statistic, 65.59, decimal=1)
    assert_almost_equal(result.pvalue, 0.999, decimal=2)
    assert result.additional_information["dof"] == 131


@pytest.mark.parametrize(
    "method",
    [
        "pearson",  # chi-square
        "log-likelihood",  # G^2
        "freeman-tukey",  # freeman-tukey
        "mod-log-likelihood",  # Modified log-likelihood
        "neyman",  # Neyman
        "cressie-read",  # Cressie-read
    ],
)
def test_chisquare_when_dependent_given_different_methodon_testdata(method):
    assert (
        power_divergence.ind(X=df_adult["Age"], Y=df_adult["Immigrant"], method=method).pvalue
        < 0.05
    )

    assert power_divergence.ind(X=df_adult["Age"], Y=df_adult["Race"], method=method).pvalue < 0.05

    assert power_divergence.ind(X=df_adult["Age"], Y=df_adult["Sex"], method=method).pvalue < 0.05
    assert (
        power_divergence.ind(X=df_adult["Immigrant"], Y=df_adult["Sex"], method=method).pvalue
        >= 0.05
    )

    assert (
        power_divergence.condind(
            X=df_adult["Education"],
            Y=df_adult["HoursPerWeek"],
            condition_on=df_adult[["Age", "Immigrant", "Race", "Sex"]],
            method=method,
        ).pvalue
        < 0.05
    )
    assert (
        power_divergence.condind(
            X=df_adult["Education"],
            Y=df_adult["MaritalStatus"],
            condition_on=df_adult[["Age", "Sex"]],
            method=method,
        ).pvalue
        < 0.05
    )


@pytest.mark.parametrize(
    "method",
    [
        "pearson",  # chi-square
        "log-likelihood",  # G^2
        "freeman-tukey",  # freeman-tukey
        "mod-log-likelihood",  # Modified log-likelihood
        "neyman",  # Neyman
        "cressie-read",  # Cressie-read
    ],
)
def test_chisquare_when_exactly_dependent_given_different_method(method):
    x = np.random.choice([0, 1], size=1000)
    y = x.copy()
    df = pd.DataFrame({"x": x, "y": y})

    result = power_divergence.ind(X=df["x"], Y=df["y"], method=method)
    assert result.additional_information["dof"] == 1
    assert_almost_equal(result.pvalue, 0, decimal=5)


@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.str_, np.uint, np.uint16, np.uint32, np.uintp]
)
def test_input_dtypes_gets_properly_computed(dtype):
    x = np.random.choice([0, 1], size=1000).astype(dtype)
    y = x.copy()
    df = pd.DataFrame({"x": x, "y": y})

    result = power_divergence.ind(X=df["x"], Y=df["y"])
    assert result.additional_information["dof"] == 1
    assert_almost_equal(result.pvalue, 0, decimal=5)


def test_g_discrete():
    """Test G^2 test for discrete data."""
    dm = np.array([testdata.dis_data]).reshape((10000, 5))
    x = 0
    y = 1
    df = pd.DataFrame.from_records(dm)

    sets = [[], [2], [2, 3], [3, 4], [2, 3, 4]]
    for idx in range(len(sets)):
        if idx == 0:
            result = power_divergence.ind(X=df[x], Y=df[y], method="log-likelihood")
        else:
            result = power_divergence.condind(
                X=df[x], Y=df[y], condition_on=df[sets[idx]], method="log-likelihood"
            )
        p = result.pvalue
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
    sets = [[2, 3, 4, 5, 6, 7]]
    with pytest.warns(UserWarning, match="Not enough samples"):
        power_divergence.condind(
            X=df[x], Y=df[y], condition_on=df[sets[0]], method="log-likelihood"
        )


def test_g_binary():
    """Test G^2 test for binary data."""
    dm = np.array([testdata.bin_data]).reshape((5000, 5))
    x = 0
    y = 1
    df = pd.DataFrame.from_records(dm)

    sets = [[], [2], [2, 3], [3, 4], [2, 3, 4]]
    for idx in range(len(sets)):
        if idx == 0:
            # for set == []
            result = power_divergence.ind(X=df[x], Y=df[y], method="log-likelihood")
        else:
            result = power_divergence.condind(
                X=df[x], Y=df[y], condition_on=df[sets[idx]], method="log-likelihood"
            )
        p = result.pvalue
        fr_p = frexp(p)
        # fr_a = frexp(testdata.bin_answer[idx])
        # assert_almost_equal(fr_p[1], fr_a[1], decimal=-1)
        assert fr_p[0] >= 0
        # assert round(fr_p[0] - fr_a[0], 4) == 0
        # assert fr_p[0] > 0

    # check error message for number of samples
    dm = np.array([testdata.bin_data]).reshape((500, 50))
    df = pd.DataFrame.from_records(dm)
    sets = [[2, 3, 4, 5, 6, 7, 8]]
    with pytest.warns(UserWarning, match="Not enough samples"):
        power_divergence.condind(
            X=df[x], Y=df[y], condition_on=df[sets[0]], method="log-likelihood"
        )


def test_g_binary_simulation():
    """Test G^2 test for binary data."""
    rng = np.random.default_rng(seed)
    n_samples = 500
    df = _binary_scm(n_samples=n_samples)
    for i in range(10):
        df[i] = rng.binomial(1, p=0.5, size=n_samples)

    result = power_divergence.ind(X=df["x"], Y=df["y"], method="log-likelihood")
    assert result.pvalue < 0.05
    result = power_divergence.ind(X=df["x1"], Y=df["y"], method="log-likelihood")
    assert result.pvalue < 0.05
    result = power_divergence.ind(X=df["x"], Y=df["x1"], method="log-likelihood")
    assert result.pvalue > 0.05
    result = power_divergence.ind(X=df["x1"], Y=df[0], method="log-likelihood")
    assert result.pvalue > 0.05

    result = power_divergence.condind(
        X=df["x"], Y=df["x1"], condition_on=df["y"], method="log-likelihood"
    )
    assert result.pvalue < 0.05


def test_g_binary_highdim():
    """Test G^2 test for binary data."""
    rng = np.random.default_rng(seed)
    n_samples = 1000
    df = _binary_scm(n_samples=n_samples)
    for i in range(10):
        df[i] = rng.binomial(1, p=0.8, size=n_samples)

    result = power_divergence.condind(
        X=df["x"], Y=df["x1"], condition_on=df[list(range(6))], method="log-likelihood"
    )
    assert result.pvalue > 0.05
    result = power_divergence.condind(
        X=df["x"], Y=df["y"], condition_on=df[list(range(5)) + ["x1"]], method="log-likelihood"
    )
    assert result.pvalue < 0.05


class TestPreprocessInputs:
    """Test error cases in preprocessing categorical data input."""

    # Test for a valid case with mixed integer and string input arrays
    def test_preprocess_inputs_mixed(self):
        X = np.array([1, "red", 3, "green"], dtype="object")
        Y = np.array(["blue", 5, "yellow", 7], dtype="object")
        with pytest.raises(TypeError):
            power_divergence.ind(X, Y)

    # Test for invalid case with 2D array as input
    def test_preprocess_inputs_2d(self):
        X = np.array([[1, 2], [3, 4]])
        Y = np.array([[5, 6], [7, 8]])
        Z = None
        with pytest.raises(ValueError):
            power_divergence.condind(X, Y, Z)

    # Test for invalid case with unsupported data type in X array
    def test_preprocess_inputs_invalid_X_dtype(self):
        X = np.array([1, "red", 3, 4.5], dtype="object")
        Y = np.array(["blue", "green", "yellow", "orange"])
        with pytest.raises(TypeError):
            power_divergence.ind(X, Y)

    # Test for invalid case with unsupported data type in Y array
    def test_preprocess_inputs_invalid_Y_dtype(self):
        X = np.array([1, 2, 3, 4])
        Y = np.array(["blue", 5, "yellow", 7.5], dtype="object")
        with pytest.raises(TypeError):
            power_divergence.ind(X, Y)
