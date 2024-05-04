"""KSample testing for equality of (conditional) distributions.

If the distributions are marginal distributions being compared, then
the test is a standard two-sample test, where the KS statistic, or
Mann-Whitney U statistic, is used to test for equality of distributions.

If the distributions are conditional distributions being compared, then
the test is a conditional two-sample test, where the KS statistic, or
Mann-Whitney U statistic, is used to test for equality of the
residual distributions, where the residuals are computed by regressing
the target variable, Y, on the conditioning variable, X.

The test statistic is described fully in :footcite:`peters2016causal`
and :footcite:`shah2018goodness`.
"""

import numpy as np
from scipy.stats import kstest

from .pvalue_result import PValueResult


def ksample(Y, Z):
    stat, pval = kstest(Y[Z == 1], Y[Z == 0])

    return PValueResult(pvalue=pval, statistic=stat)


def condksample(Y, Z, X, residual_test="ks", target_predictor=None, combine_pvalues=True):
    r"""
    Calulates the 2-sample test statistic.

    Parameters
    ----------
    Y : ndarray, shape (n_samples,)
        Target or outcome features
    X : ndarray, shape (n_samples, n_features)
        Features to condition on
    Z : list or ndarray, shape (n_samples,)
        List of zeros and ones indicating which samples belong to
        which groups.
    target_predictor : sklearn.BaseEstimator, default=None
        Method to predict the target given the covariates. If None,
        uses a spline regression with 4 knots and degree 3 as
        described in :footcite:`peters2016causal`.
    residual_test : {"whitney_levene", "ks"}, default="ks"
        Test of the residuals between the groups
    combine_pvalues: bool, default=True
        If True, returns hte minimum of the corrected pvalues.

    Returns
    -------
    pvalue : float
        The computed *k*-sample p-value.
    r2 : float
        r2 score of the regression fit
    model : object
        Fitted regresion model, if return_model is True
    """
    from sklearn.metrics import r2_score

    if target_predictor is None:
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import SplineTransformer

        pipe = Pipeline(
            steps=[
                ("spline", SplineTransformer(n_knots=4, degree=3)),
                ("linear", LinearRegression()),
            ]
        )
        param_grid = {
            "spline__n_knots": [3, 5, 7, 9],
        }
        target_predictor = GridSearchCV(
            pipe, param_grid, n_jobs=-2, refit=True, scoring="neg_mean_squared_error"
        )

    target_predictor.fit(X, Y)
    Y_pred = target_predictor.predict(X)
    residuals = Y - Y_pred
    r2 = r2_score(Y, Y_pred)

    if residual_test == "whitney_levene":
        from scipy.stats import levene, mannwhitneyu

        _, mean_pval = mannwhitneyu(
            residuals[np.asarray(Z, dtype=bool)],
            residuals[np.asarray(1 - Z, dtype=bool)],
        )
        _, var_pval = levene(
            residuals[np.asarray(Z, dtype=bool)],
            residuals[np.asarray(1 - Z, dtype=bool)],
        )
        # Correct for multiple tests
        if combine_pvalues:
            pval = min(mean_pval * 2, var_pval * 2, 1)
        else:
            pval = (min(mean_pval * 2, 1), min(var_pval * 2, 1))
    elif residual_test == "ks":
        from scipy.stats import kstest

        _, pval = kstest(
            residuals[np.asarray(Z, dtype=bool)],
            residuals[np.asarray(1 - Z, dtype=bool)],
        )
    else:
        raise ValueError(f"Test {residual_test} not a valid option.")

    return PValueResult(
        statistic=r2, pvalue=pval, additional_info={"target_predictor": target_predictor}
    )
