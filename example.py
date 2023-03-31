import numpy as np
import pandas as pd

from cmi import cmi
from independence_test import independence_test, Methods

Z = np.random.normal(0, 1, 1000)
X = Z + np.random.normal(0, 1, 1000)
Y = Z + np.random.normal(0, 1, 1000)

# Use simplified function
print(independence_test(X, Y))
print(independence_test(X, Y, condition_on=Z))

# We could still customize parameters
print(independence_test(X, Y, method=Methods.KCI, kernelX="Linear"))
print(independence_test(X, Y, condition_on=Z, method=Methods.KCI, kernelX="Linear"))

# Or call the functions directly - Here, the CMI method from dodsicover
print(cmi.unconditional_independence(X, Y, k=0.3))
print(cmi.conditional_independence(X, Y, condition_on=Z, n_shuffle=200))

# In case we have pandas data frames, we can equally easily use them as well and avoid enforcing some synthetic column
# names
my_data_frame = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})

print(independence_test(my_data_frame['X'], my_data_frame['Y']))
print(independence_test(my_data_frame['X'], my_data_frame['Y'], condition_on=my_data_frame['Z']))
