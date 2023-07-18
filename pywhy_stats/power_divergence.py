"""Independence test among categorical variables using power-divergence tests.

Works on categorical random variables. Based on the ``method`` parameter, one
can compute a wide variety of different categorical hypothesis tests.

Categorical data is a type of data that can be divided into discrete groups.
Compared to continuous data, there is no agreed upon way to represent
categorical data numerically. For example, we can represent the color of an
object as "red", "blue", "green" etc. but we can also represent it as 1, 2, 3,
which maps to those colors, or even [1.2, 2.2, 3.2] which maps to those colors.

If `str` type data is passed in, then it is converted to `int` type using
`sklearn.preprocessing.LabelEncoder`. All columns of data passed in must be of
the same type, otherwise it is impossible to infer what you want to do.

Encoding categorical data numerically is a common practice in machine learning
and statistics. There are many strategies, and we do not implement
most of them. For categorical encoding strategies, see
https://github.com/scikit-learn-contrib/category_encoders.

Examples
--------
>>> import pywhy_stats as ps
>>> res = ps.categorical.ind([1, 2, 3], [4, 5, 6])
>>> print(res.pvalue)
>>> 1.0
"""

import logging
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from sklearn.preprocessing import LabelEncoder

from .pvalue_result import PValueResult


def ind(
    X: ArrayLike, Y: ArrayLike, method: str = "cressie-read", num_categories_allowed: int = 10
) -> PValueResult:
    """Perform an independence test using power divergence test.

    The null hypothesis for the test is X is independent of Y. A lot of the
    frequency comparison based statistics (eg. chi-square, G-test etc) belong to
    power divergence family, and are special cases of this test.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples,)
        The first node variable.
    Y : ArrayLike of shape (n_samples,)
        The second node variable.
    method : float or string
        The lambda parameter for the power_divergence statistic. Some values of
        ``method`` results in other well known tests:

            "pearson"             1          "Chi-squared test"
            "log-likelihood"      0          "G-test or log-likelihood"
            "freeman-tukey"     -1/2         "freeman-tukey Statistic"
            "mod-log-likelihood"  -1         "Modified Log-likelihood"
            "neyman"              -2         "Neyman's statistic"
            "cressie-read"        2/3        "The value recommended in the paper :footcite:`cressieread1984`"
    num_categories_allowed : int
        The maximum number of categories allowed in the input variables. Default
        of 10 is chosen to error out on large number of categories.

    Returns
    -------
    statistic : float
        The test statistic.
    pvalue : float
        The p-value of the test.

    References
    ----------
    .. footbibliography::
    """
    X, Y, _ = _preprocess_inputs(X=X, Y=Y, Z=None)
    return _power_divergence(
        X=X, Y=Y, Z=None, method=method, num_categories_allowed=num_categories_allowed
    )


def condind(
    X: ArrayLike,
    Y: ArrayLike,
    condition_on: ArrayLike,
    method: str = "cressie-read",
    num_categories_allowed: int = 10,
) -> PValueResult:
    """Perform an independence test using power divergence test.

    The null hypothesis for the test is X is independent of Y given condition_on.
    A lot of the frequency comparison based statistics (eg. chi-square, G-test etc)
    belong to power divergence family, and are special cases of this test.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples,)
        The first node variable.
    Y : ArrayLike of shape (n_samples,)
        The second node variable.
    condition_on : ArrayLike of shape (n_samples, n_variables)
        The conditioning set.
    method : float or string
        The lambda parameter for the power_divergence statistic. Some values of
        method results in other well known tests:

            "pearson"             1          "Chi-squared test"
            "log-likelihood"      0          "G-test or log-likelihood"
            "freeman-tukey"     -1/2         "freeman-tukey Statistic"
            "mod-log-likelihood"  -1         "Modified Log-likelihood"
            "neyman"              -2         "Neyman's statistic"
            "cressie-read"        2/3        "The value recommended in the paper :footcite:`cressieread1984`"
    num_categories_allowed : int
        The maximum number of categories allowed in the input variables. Default
        of 10 is chosen to error out on large number of categories.

    Returns
    -------
    statistic : float
        The test statistic.
    pvalue : float
        The p-value of the test.
    """
    X, Y, condition_on = _preprocess_inputs(X=X, Y=Y, Z=condition_on)
    return _power_divergence(
        X=X, Y=Y, Z=condition_on, method=method, num_categories_allowed=num_categories_allowed
    )


def _preprocess_inputs(X: ArrayLike, Y: ArrayLike, Z: Optional[ArrayLike]) -> ArrayLike:
    """Preprocess inputs for categorical independence tests.

    Returns
    -------
    X, Y, Z : ArrayLike of shape (n_samples, [n_conditions]) of type np.int
        The preprocessed inputs.
    """
    if X.ndim != 1:
        if X.shape[1] == 1:
            X = X.reshape(-1)
        else:
            raise ValueError("X should be 1-D arrays.")
    if Y.ndim != 1:
        if Y.shape[1] == 1:
            Y = Y.reshape(-1)
        else:
            raise ValueError("Y should be 1-D arrays.")

    # ensure we use numpy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)

    if not all(type(xi) == type(X[0]) for xi in X):  # noqa
        raise ValueError("All elements of X must be of the same type.")
    if not all(type(yi) == type(Y[0]) for yi in Y):  # noqa
        raise ValueError("All elements of Y must be of the same type.")

    # Check if all elements are integers
    if np.issubdtype(type(X[0]), np.str_):
        le = LabelEncoder()
        X = le.fit_transform(X)
        # warn("Converting X array to categorical array using scikit-learn's LabelEncoder.")
    elif not np.issubdtype(type(X[0]), np.integer):
        raise TypeError(
            f"X should be an array of integers (np.integer), or strings (np.str_), not {type(X[0])}."
        )

    # Ensure now all elements are integers
    if np.issubdtype(type(Y[0]), np.str_):
        le = LabelEncoder()
        Y = le.fit_transform(Y)
        # warn("Converting Y array to categorical array using scikit-learn's LabelEncoder.")
    elif not np.issubdtype(type(Y[0]), np.integer):
        raise TypeError(
            f"Y should be an array of integers (np.integer), or strings (np.str_), not {type(Y[0])}."
        )

    if Z is not None:
        Z = np.asarray(Z)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        for icol in range(Z.shape[1]):
            if not all(type(zi) == type(Z[0, icol]) for zi in Z[:, icol]):  # noqa
                raise ValueError(f"All elements of Z in column {icol} must be of the same type.")

            # XXX: needed when converting to only numpy API
            # Check if all elements are integers
            if np.issubdtype(type(Z[0, icol]), np.str_):
                le = LabelEncoder()
                Z[:, icol] = le.fit_transform(Z[:, icol])
                # warn("Converting Z array to categorical array using scikit-learn's LabelEncoder.")
            elif not np.issubdtype(type(Z[0, icol]), np.integer):
                raise TypeError(
                    f"Z should be an array of integers (np.integer), or strings (np.str_), not {type(Z[0, icol])}."
                )
    return X, Y, Z


# This is a modified function taken from pgmpy: License MIT
def _power_divergence(
    X: ArrayLike,
    Y: ArrayLike,
    Z: Optional[ArrayLike],
    method: str = "cressie-read",
    num_categories_allowed: int = 10,
) -> PValueResult:
    """Compute the Cressie-Read power divergence statistic.

    Parameters
    ----------
    X: ArrayLike of shape (n_samples,) of type np.int
        The first node variable.
    Y : ArrayLike of shape (n_samples,) of type np.int
        The second node variable.
    Z : optional, ArrayLike of shape (n_samples, n_variables) of type np.int
        The conditioning set. If not defined, is `None`.
    method : float or string
        The lambda parameter for the power_divergence statistic. Some values of
        method results in other well known tests:

        "pearson"             1          "Chi-squared test"
        "log-likelihood"      0          "G-test or log-likelihood"
        "freeman-tukey"     -1/2         "freeman-tukey Statistic"
        "mod-log-likelihood"  -1         "Modified Log-likelihood"
        "neyman"              -2         "Neyman's statistic"
        "cressie-read"        2/3        "The value recommended in the paper
                                         :footcite:`cressieread1984`"
    num_categories_allowed : int
        The maximum number of categories allowed in the input variables.

    Returns
    -------
    CI Test Results: tuple
        Returns a tuple (chi, p_value, dof). `chi` is the
        chi-squared test statistic. The `p_value` for the test, i.e. the
        probability of observing the computed chi-square statistic (or an even
        higher value), given the null hypothesis that X \u27C2 Y | Zs is True.
        If boolean = True, returns True if the p_value of the test is greater
        than `significance_level` else returns False.

    See Also
    --------
    scipy.stats.power_divergence

    References
    ----------
    .. footbibliography::
    """
    for name, arr in zip(["X", "Y"], [X, Y]):
        # Check if the number of unique values is reasonably small
        unique_values = np.unique(arr)
        num_unique_values = len(unique_values)
        # XXX: We chose some arbitrary value here. Adjust the threshold as needed based on user feedback.
        if num_unique_values > num_categories_allowed:
            raise RuntimeError(
                f"There are {num_unique_values} > {num_categories_allowed} unique categories for {name}. "
                f"This is likely an error."
            )

    # Step 1: Do a simple contingency test if there are no conditional variables.
    if Z is None:
        # Compute the contingency table
        observed_xy, _, _ = np.histogram2d(X, Y, bins=(np.unique(X).size, np.unique(Y).size))
        chi, p_value, dof, expected = stats.chi2_contingency(observed_xy, method=method)

    # Step 2: If there are conditionals variables, iterate over unique states and do
    #         the contingency test.
    else:
        import pandas as pd

        chi = 0
        dof = 0

        # check number of samples relative to degrees of freedom
        # assuming no zeros
        s_size = Z.shape[1]
        n_samples = Z.shape[0]

        # Rule of thumb: need at least 10 samples per degree of freedom (dof)
        levels_x = len(np.unique(X))
        levels_y = len(np.unique(Y))
        dof_check = (
            (levels_x - 1)
            * (levels_y - 1)
            * np.prod([len(np.unique(Z[:, i])) for i in range(s_size)])
        )
        n_samples_req = 10 * dof_check
        if n_samples < n_samples_req:
            raise RuntimeError(
                f"Not enough samples. {n_samples} is too small. Need {n_samples_req}."
            )

        # XXX: currently we just leverage pandas to do the grouping. This is not
        # ideal since we do not want the reliance on pandas package, but we should refactor
        # this to only use numpy efficiently
        X_columns = ["X"]
        Y_columns = ["Y"]
        Z_columns = [f"Z{i}" for i in range(Z.shape[1])]
        columns = X_columns + Y_columns + Z_columns
        data = pd.DataFrame(np.column_stack((X, Y, Z)), columns=columns)

        for z_state, df in data.groupby(Z_columns[0] if len(Z_columns) == 1 else Z_columns):
            try:
                # Note: The fill value is set to 1e-7 to avoid the following error:
                # where there are not enough samples in the data, which results in a nan pvalue
                sub_table_z = (
                    df.groupby(X_columns + Y_columns).size().unstack(Y_columns, fill_value=1e-7)
                )
                c, _, d, _ = stats.chi2_contingency(sub_table_z, method=method)
                chi += c
                dof += d
            except ValueError:
                # If one of the values is 0 in the 2x2 table.
                if isinstance(z_state, str):
                    logging.info(f"Skipping the test X \u27C2 Y | Z={z_state}. Not enough samples")
                else:
                    z_str = ", ".join([f"{var}={state}" for var, state in zip(Z_columns, z_state)])
                    logging.info(f"Skipping the test X \u27C2 Y | {z_str}. Not enough samples")

            if np.isnan(c):
                raise RuntimeError(
                    f"The resulting chi square test statistic is NaN, which occurs "
                    f"when there are not enough samples in your data "
                    f"{df.shape}, {sub_table_z}."
                )

        p_value = 1 - stats.chi2.cdf(chi, df=dof)

    # Step 4: Return the values
    additional_information = {"dof": dof}
    result = PValueResult(
        statistic=chi, pvalue=p_value, additional_information=additional_information
    )
    return result
