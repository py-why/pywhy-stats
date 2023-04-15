from enum import Enum
from types import ModuleType
from typing import Optional
from warnings import warn

import scipy.stats
from numpy.typing import ArrayLike

from pywhy_stats import fisherz

from .pvalue_result import PValueResult


class Methods(Enum):
    """Methods for independence testing."""

    AUTO = 0
    """Choose an automatic method based on the data."""

    FISHERZ = fisherz
    """:py:mod:`~pywhy_stats.fisherz`: Fisher's Z test for independence"""


def independence_test(
    X: ArrayLike,
    Y: ArrayLike,
    condition_on: Optional[ArrayLike] = None,
    method=Methods.AUTO,
    **kwargs,
) -> PValueResult:
    """Perform a (conditional) independence test to determine whether X and Y are independent.

    The test may be conditioned on an optional set of variables. This is, test whether ``X _||_ Y |
    condition_on``, where the null hypothesis is that X and Y are independent.

    Parameters
    ----------
    X : ArrayLike, shape (n_samples, n_features_x)
        Data matrix for X.
    Y : ArrayLike, shape (n_samples, n_features_y)
        Data matrix for Y.
    condition_on : ArrayLike or None, shape (n_samples, n_features_z), optional
        Data matrix for the conditioning variables. If None is given, an unconditional test
        is performed.
    method : Methods, optional
        Independence test method from the Methods enum. Default is Methods.AUTO, which will
        automatically select an appropriate method.
    **kwargs : dict or None, optional
        Additional keyword arguments to be passed to the specific test method

    Returns
    -------
    result : PValueResult
        An instance of the PValueResult data class, containing the p-value, test statistic,
        and any additional information related to the independence test.

    See Also
    --------
    fisherz : Fisher's Z test for independence
    """
    method_func: ModuleType
    if method == Methods.AUTO:
        method_func = Methods.FISHERZ
    else:
        method_func = method

    if method_func == Methods.FISHERZ:
        if condition_on is None:
            data = [X, Y]
        else:
            data = [X, Y, condition_on]
        for _data in data:
            _, pval = scipy.stats.normaltest(_data)

            # XXX: we should add pinguoin as an optional dependency for doing multi-comp stuff
            if pval < 0.05:
                warn("Your data is not all normal but you're trying to use partial correlation")

    if condition_on is None:
        return method_func.ind(X, Y, method, **kwargs)
    else:
        return method_func.condind(X, Y, condition_on, method, **kwargs)
