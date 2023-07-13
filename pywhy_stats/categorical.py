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

from .pvalue_result import PValueResult


def ind(X: ArrayLike, Y: ArrayLike, lambda_: str = "cressie-read") -> PValueResult:
    """Perform an independence test using power divergence test.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples,)
        The first node variable.
    Y : ArrayLike of shape (n_samples,)
        The second node variable.
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


#    # Step 1: Check if the arguments are valid and type conversions.
#     if isinstance(Z, str):
#         Z = [Z]
#     if (X in Z) or (Y in Z):
#         raise ValueError(f"The variables X or Y can't be in Z. Found {X if X in Z else Y} in Z.")


# This is a modified function taken from pgmpy: License MIT
def _power_divergence(
    X: ArrayLike, Y: ArrayLike, Z: Optional[ArrayLike], lambda_: str = "cressie-read"
) -> PValueResult:
    """Computes the Cressie-Read power divergence statistic [1].

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

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
    >>> data['E'] = data['A'] + data['B'] + data['C']
    >>> chi_square(X='A', Y='C', Z=[], data=data, boolean=True, significance_level=0.05)
    True
    >>> chi_square(X='A', Y='B', Z=['D'], data=data, boolean=True, significance_level=0.05)
    True
    >>> chi_square(X='A', Y='B', Z=['D', 'E'], data=data, boolean=True, significance_level=0.05)
    False
    """
    # Step 1: Do a simple contingency test if there are no conditional variables.
    if Z is None:
        # XXX: convert this to work with numpy arrays
        chi, p_value, dof, expected = stats.chi2_contingency(
            data.groupby([X, Y]).size().unstack(Y, fill_value=0), lambda_=lambda_
        )

    # Step 2: If there are conditionals variables, iterate over unique states and do
    #         the contingency test.
    else:
        chi = 0
        dof = 0
        for idx, (z_state, df) in enumerate(data.groupby(Z[0] if len(Z) == 1 else Z)):
            try:
                # Note: The fill value is set to 1e-7 to avoid the following error:
                # where there are not enough samples in the data, which results in a nan pvalue
                sub_table_z = df.groupby([X, Y]).size().unstack(Y, fill_value=1e-7)
                c, _, d, _ = stats.chi2_contingency(sub_table_z, lambda_=lambda_)
                chi += c
                dof += d
            except ValueError:
                # If one of the values is 0 in the 2x2 table.
                if isinstance(z_state, str):
                    logging.info(
                        f"Skipping the test {X} \u27C2 {Y} | {Z[idx]}={z_state}. Not enough samples"
                    )
                else:
                    z_str = ", ".join([f"{var}={state}" for var, state in zip(Z, z_state)])
                    logging.info(f"Skipping the test {X} \u27C2 {Y} | {z_str}. Not enough samples")

            if np.isnan(c):
                raise RuntimeError(
                    f"The resulting chi square test statistic is NaN, which occurs "
                    f"when there are not enough samples in your data "
                    f"{df.shape}, {sub_table_z}."
                )

        p_value = 1 - stats.chi2.cdf(chi, df=dof)

    # Step 4: Return the values
    result = PValueResult(statistic=chi, pvalue=p_value, dof=dof)
    return result
