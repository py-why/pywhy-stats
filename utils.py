import numpy as np


def shape_into_2d(*args):
    """If necessary, shapes the numpy inputs into 2D matrices.

    Example:
        array([1, 2, 3]) -> array([[1], [2], [3]])
        2 -> array([[2]])

    :param args: The function expects numpy arrays as inputs and returns a reshaped (2D) version of them (if necessary).
    :return: Reshaped versions of the input numpy arrays. For instance, given 1D inputs X, Y and Z, then
             shape_into_2d(X, Y, Z) reshapes them into 2D and returns them. If an input is already 2D, it will not be
             modified and returned as it is.
    """

    def shaping(X: np.ndarray):
        if X.ndim < 2:
            return np.column_stack([X])
        elif X.ndim > 2:
            raise ValueError("Cannot reshape a %dD array into a 2D array!" % X.ndim)

        return X

    result = [shaping(x) for x in args]

    if len(result) == 1:
        return result[0]
    else:
        return result