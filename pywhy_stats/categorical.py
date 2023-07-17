"""Independence test among categorical variables using power-divergence tests.

Works on categorical random variables. Based on the ``lambda_`` parameter, one
can compute a wide variety of different categorical hypothesis tests.

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


def ind(X: ArrayLike, Y: ArrayLike, lambda_: str = "cressie-read") -> PValueResult:
    """Perform an independence test using power divergence test.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples,)
        The first node variable.
    Y : ArrayLike of shape (n_samples,)
        The second node variable.
    lambda_ : float or string
        The lambda parameter for the power_divergence statistic. Some values of
        ``lambda_`` results in other well known tests:
            "pearson"             1          "Chi-squared test"
            "log-likelihood"      0          "G-test or log-likelihood"
            "freeman-tukey"     -1/2        "freeman-tukey Statistic"
            "mod-log-likelihood"  -1         "Modified Log-likelihood"
            "neyman"              -2         "Neyman's statistic"
            "cressie-read"        2/3        "The value recommended in the paper
                                             :footcite:`cressieread1984`"

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
    return _power_divergence(X=X, Y=Y, Z=None, lambda_=lambda_)


def condind(
    X: ArrayLike, Y: ArrayLike, condition_on: ArrayLike, lambda_: str = "cressie-read"
) -> PValueResult:
    """Perform an independence test using power divergence test.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples,)
        The first node variable.
    Y : ArrayLike of shape (n_samples,)
        The second node variable.
    condition_on : ArrayLike of shape (n_samples, n_variables)
        The conditioning set.
    lambda_: float or string
        The lambda parameter for the power_divergence statistic. Some values of
        lambda_ results in other well known tests:
            "pearson"             1          "Chi-squared test"
            "log-likelihood"      0          "G-test or log-likelihood"
            "freeman-tukey"     -1/2        "freeman-tukey Statistic"
            "mod-log-likelihood"  -1         "Modified Log-likelihood"
            "neyman"              -2         "Neyman's statistic"
            "cressie-read"        2/3        "The value recommended in the paper
                                             :footcite:`cressieread1984`"

    Returns
    -------
    statistic : float
        The test statistic.
    pvalue : float
        The p-value of the test.
    """
    return _power_divergence(X=X, Y=Y, Z=condition_on, lambda_=lambda_)


def _compute_conditional_contingency_table(X, Y, Z):
    unique_Z_vals = np.unique(Z)

    chi = 0
    dof = 0

    for z_state in unique_Z_vals:
        pass


def _compute_conditional_contingency_table(df, X, Y, Z, lambda_=None):
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    unique_Z_vals = np.unique(Z)

    chi = 0
    dof = 0

    for z_state in unique_Z_vals:
        z_indices = np.where(Z == z_state)[0]
        sub_df = df[z_indices]

        x_values = np.unique(sub_df[X])
        y_values = np.unique(sub_df[Y])

        counts, _, _ = np.histogram2d(sub_df[X], sub_df[Y], bins=(x_values, y_values))

        if lambda_ is not None:
            counts = lambda_(counts)

        c, _ = stats.chisquare(counts)
        chi += c
        dof += (counts.shape[0] - 1) * (counts.shape[1] - 1)

    return chi, dof


# This is a modified function taken from pgmpy: License MIT
def _power_divergence(
    X: ArrayLike, Y: ArrayLike, Z: Optional[ArrayLike], lambda_: str = "cressie-read"
) -> PValueResult:
    """Compute the Cressie-Read power divergence statistic.

    The null hypothesis for the test is X is independent of Y given Z. A lot of the
    frequency comparison based statistics (eg. chi-square, G-test etc) belong to
    power divergence family, and are special cases of this test.

    Parameters
    ----------
    X: ArrayLike of shape (n_samples,)
        The first node variable.
    Y : ArrayLike of shape (n_samples,)
        The second node variable.
    Z : ArrayLike of shape (n_samples, n_variables)
        The conditioning set.
    lambda_: float or string
        The lambda parameter for the power_divergence statistic. Some values of
        lambda_ results in other well known tests:
            "pearson"             1          "Chi-squared test"
            "log-likelihood"      0          "G-test or log-likelihood"
            "freeman-tukey"     -1/2        "freeman-tukey Statistic"
            "mod-log-likelihood"  -1         "Modified Log-likelihood"
            "neyman"              -2         "Neyman's statistic"
            "cressie-read"        2/3        "The value recommended in the paper
                                             :footcite:`cressieread1984`"

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
    if X.ndim != 1 or Y.ndim != 1:
        raise ValueError("X and Y should be 1-D arrays.")

    # ensure we use numpy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Check if all elements are integers
    if not np.issubdtype(X.dtype, np.integer):
        le = LabelEncoder()
        X = le.fit_transform(X)
        # warn("Converting X array to categorical array using scikit-learn's LabelEncoder.")

    # Check if all elements are integers
    if not np.issubdtype(Y.dtype, np.integer):
        le = LabelEncoder()
        Y = le.fit_transform(Y)
        # warn("Converting Y array to categorical array using scikit-learn's LabelEncoder.")

    for name, arr in zip(["X", "Y"], [X, Y]):
        # Check if the number of unique values is reasonably small
        unique_values = np.unique(arr)
        num_unique_values = len(unique_values)
        if num_unique_values > 64:  # Adjust the threshold as needed
            raise RuntimeError(
                f"There are {num_unique_values} unique categories for {name}. "
                f"This is likely an error."
            )

    # Step 1: Do a simple contingency test if there are no conditional variables.
    if Z is None:
        # Compute the contingency table
        observed_xy, _, _ = np.histogram2d(X, Y, bins=(np.unique(X).size, np.unique(Y).size))
        chi, p_value, dof, expected = stats.chi2_contingency(observed_xy, lambda_=lambda_)

    # Step 2: If there are conditionals variables, iterate over unique states and do
    #         the contingency test.
    else:
        import pandas as pd

        chi = 0
        dof = 0

        Z = np.asarray(Z)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        # XXX: needed when converting to only numpy API
        # Check if all elements are integers
        if not np.issubdtype(Z.dtype, np.integer):
            le = LabelEncoder()
            for idx in range(Z.shape[1]):
                Z[:, idx] = le.fit_transform(Z[:, idx])
            # warn("Converting Z array to categorical array using scikit-learn's LabelEncoder.")

        # check number of samples relative to degrees of freedom
        # assuming no zeros
        s_size = Z.shape[1]
        n_samples = Z.shape[0]

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
                c, _, d, _ = stats.chi2_contingency(sub_table_z, lambda_=lambda_)
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
