from enum import Enum
from types import ModuleType
from typing import Callable, Optional

from numpy.testing import ArrayLike

from pywhy_stats import fisherz

from .p_value_result import PValueResult


class Methods(Enum):
    """Methods for independence testing."""

    AUTO = 0
    FISHERZ = fisherz


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

    if condition_on is None:
        return method_func.ind(X, Y, method, **kwargs)
    else:
        return method_func.condind(X, Y, condition_on, method, **kwargs)
