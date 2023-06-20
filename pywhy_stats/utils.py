import inspect
import random
from typing import Callable

import numpy as np


def preserve_random_state(func: Callable) -> Callable:
    """
    Decorate function for setting and preserving the state of random generators before and after calling it.

    The decorated function is expected to have a 'random_seed' parameter which is used to set a new seed for the
    duration of the function call.

    Parameters
    ----------
    func : Callable
        The function to be decorated.

    Returns
    -------
    Callable
        The decorated function which saves the random state before executing the
        input function, sets a new random seed, then restores the original random
        state after executing the function.
    """
    sig = inspect.signature(func)

    def wrapper(*args, **kwargs):
        # Get bound arguments
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        if "random_seed" not in bound_args.arguments:
            raise ValueError(
                "'random_seed' parameter is missing for use with the random state decorator!"
            )

        # Original random states
        numpy_state = np.random.get_state()
        random_state = random.getstate()

        if bound_args.arguments["random_seed"] is not None:
            np.random.seed(kwargs["random_seed"])
            random.seed(kwargs["random_seed"])

            try:
                result = func(*args, **kwargs)
            finally:
                # Restore the original random state
                np.random.set_state(numpy_state)
                random.setstate(random_state)

            return result
        else:
            return func(*args, **kwargs)

    return wrapper
