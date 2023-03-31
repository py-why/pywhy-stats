from enum import Enum
from typing import Optional

import numpy as np

from analysis_result import AnalysisResult
from cmi import cmi
from kci import kci


class Methods(Enum):
    AUTO = 0
    KCI = 1,
    CMI = 2


def independence_test(X: np.ndarray,
                      Y: np.ndarray,
                      condition_on: Optional[np.ndarray] = None,
                      method=Methods.AUTO,
                      **kwargs) -> AnalysisResult:
    # Perform some checks on the data type and conclude suitable method choice
    if method == Methods.AUTO:
        method = Methods.KCI

    if condition_on is None:
        return unconditional_independence(X, Y, method, **kwargs)
    else:
        return conditional_independence(X, Y, condition_on, method, **kwargs)


def unconditional_independence(X: np.ndarray,
                               Y: np.ndarray,
                               method=Methods.KCI,
                               **kwargs) -> AnalysisResult:
    if method == Methods.KCI:
        return kci.unconditional_independence(X, Y, **kwargs)
    elif method == Methods.CMI:
        return cmi.unconditional_independence(X, Y, **kwargs)
    else:
        raise ValueError("Method not found!")


def conditional_independence(X: np.ndarray,
                             Y: np.ndarray,
                             condition_on: np.ndarray,
                             method=Methods.KCI,
                             **kwargs) -> AnalysisResult:
    if method == Methods.KCI:
        return kci.conditional_independence(X, Y, condition_on, **kwargs)
    elif method == Methods.CMI:
        return cmi.conditional_independence(X, Y, condition_on, **kwargs)
    else:
        raise ValueError("Method not found!")
