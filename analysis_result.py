from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


@dataclass
class AnalysisResult:
    # Serving as result object from the algorithms

    p_value: float
    test_statistic: Optional[Union[float, np.ndarray]] = None
    additional_information: Optional[object] = None
