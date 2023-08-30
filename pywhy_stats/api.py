from enum import Enum
from types import ModuleType
from typing import Optional
from warnings import warn

import numpy as np
import scipy.stats
from numpy.typing import ArrayLike

from pywhy_stats.independence import fisherz, kci

from .pvalue_result import PValueResult


class Methods(Enum):
    """Methods for independence testing."""

    AUTO = 0
    """Choose an automatic method based on the data."""

    FISHERZ = fisherz
    """:py:mod:`pywhy_stats.independence.fisherz`: Fisher's Z test for independence"""

    KCI = kci
    """:py:mod:`pywhy_stats.independence.kci`: Conditional kernel independence test"""


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
        Independence test method from the :class:`pywhy_stats.Methods` enum. Default is
        `Methods.AUTO`, which will automatically select an appropriate method.
    **kwargs : dict or None, optional
        Additional keyword arguments to be passed to the specific test method

    Returns
    -------
    result : PValueResult
        An instance of the PValueResult data class, containing the p-value, test statistic,
        and any additional information related to the independence test.

    See Also
    --------
    pywhy_stats.independence.fisherz : Fisher's Z test for independence
    pywhy_stats.independence.kci : Kernel Conditional Independence test
    """
    method_module: ModuleType
    if method == Methods.AUTO:
        method_module = Methods.KCI
    elif not isinstance(method, Methods):
        raise ValueError(
            f"Invalid method type. Expected one of {Methods.__members__.keys()}, "
            f"but got {method}."
        )
    else:
        method_module = method  # type: ignore

    if method_module == Methods.FISHERZ:
        if condition_on is None:
            data = [X, Y]
        else:
            data = [X, Y, condition_on]
        for _data in data:
            res = scipy.stats.normaltest(_data, axis=0)

            # XXX: we should add pinguoin as an optional dependency for doing multi-comp stuff
            if np.atleast_1d(res.pvalue).any() < 0.05:
                warn(
                    "The provided data does not seem to be Gaussian, but the Fisher-Z test "
                    "assumes that the data follows a Gaussian distribution. The result should "
                    "be interpreted carefully or consider a different independence test method."
                )

    if condition_on is None:
        return method_module.value.ind(X, Y, **kwargs)
    else:
        return method_module.value.condind(X, Y, condition_on, **kwargs)
