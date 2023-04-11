from dataclasses import dataclass
from typing import Optional, Union

from numpy.typing import ArrayLike


@dataclass
class PValueResult:
    """Data class representing the results of an hypothesis test that produces a p-value.

    Attributes
    ----------
    p_value: float
        The p-value represents the probability of observing the given test statistic, or more
        extreme results, under a certain null hypothesis.
    test_statistic: float or numpy.ndarray or None
        The test statistic of the hypothesis test, which might not always be available.
    additional_information: object or None
        Any additional information or metadata relevant to the specific test conducted. These could
        also be a state of the method to re-use it.
    """

    p_value: float
    test_statistic: Optional[Union[float, ArrayLike]] = None
    additional_information: Optional[object] = None
