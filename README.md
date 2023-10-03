[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CircleCI](https://circleci.com/gh/py-why/pywhy-stats/tree/main.svg?style=svg)](https://circleci.com/gh/py-why/pywhy-stats/tree/main)
[![unit-tests](https://github.com/py-why/pywhy-stats/actions/workflows/main.yml/badge.svg)](https://github.com/py-why/pywhy-stats/actions/workflows/main.yml)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![codecov](https://codecov.io/gh/py-why/pywhy-stats/branch/main/graph/badge.svg?token=H1reh7Qwf4)](https://codecov.io/gh/py-why/pywhy-stats)
[![PyPI Download count](https://img.shields.io/pypi/dm/pywhy-stats.svg)](https://pypistats.org/packages/pywhy-stats)
[![Latest PyPI release](https://img.shields.io/pypi/v/pywhy-stats.svg)](https://pypi.org/project/pywhy-stats/)

# PyWhy-Stats

Pywhy-stats serves as Python library for implementations of various statistical methods, such as (un)conditional independence tests, which can be utilized in tasks like causal discovery. In the current version, PyWhy-stats supports:
- Kernel-based independence and conditional k-sample tests
- FisherZ-based independence tests
- Power-divergence independence tests
- Bregman-divergence conditional k-sample tests

# Documentation

See the [development version documentation](https://py-why.github.io/pywhy-stats/dev/index.html).

Or see [stable version documentation](https://py-why.github.io/pywhy-stats/stable/index.html)

# Installation

Installation is best done via `pip` or `conda`. For developers, they can also install from source using `pip`. See [installation page](https://www.pywhy.org/pywhy-stats/dev/installation.html) for full details.

# Quick Start

In the following sections, we will use artificial exemplary data to demonstrate the API's functionality. More
information about the methods and hyperparameters can be found in the [documentation](https://py-why.github.io/pywhy-stats/stable/index.html).

Note that most methods in PyWhy-Stats support multivariate inputs. For this. simply pass in a
2D numpy array where rows represent samples and columns the different dimensions.

### Unconditional Independence Tests

Consider the following exemplary data:

```
import numpy as np
  
rng = np.random.default_rng(0)
X = rng.standard_normal((200, 1))
Y = np.exp(X + rng.standard_normal(size=(200, 1)))
```

Here, $Y$ depends on $X$ in a non-linear way. We can use the simplified API of PyWhy-Stats to test the null hypothesis
that the variables are independent:

```
from pywhy_stats import independence_test
 
result = independence_test(X, Y)
print("p-value:", result.pvalue, "Test statistic:", result.statistic)
```

The `independence_test` method returns an object containing a p-value, a test statistic, and possibly additional
information about the test. By default, this method employs a heuristic to select the most appropriate test for the
data. Currently, it defaults to a kernel-based independence test.

As we observed, the p-value is significantly small. Using, for example, a significance level of 0.05, we would reject
the null hypothesis of independence and infer that these variables are dependent. However, a p-value exceeding the
significance level doesn't conclusively indicate that the variables are independent, it only indicates insufficient
evidence of dependence.

We can also be more specific in the type of independence test we want to use. For instance, to use
a FisherZ test, we can indicate this by:

```
from pywhy_stats import Methods

result = independence_test(X, Y, method=Methods.FISHERZ)
print("p-value:", result.pvalue, "Test statistic:", result.statistic)
```

Or for the kernel based independence test:

```
from pywhy_stats import Methods

result = independence_test(X, Y, method=Methods.KCI)
print("p-value:", result.pvalue, "Test statistic:", result.statistic)
```

For more information about the available methods, hyperparameters and other details, see the
[documentation](https://py-why.github.io/pywhy-stats/stable/index.html).

### Conditional independence test

Similar to the unconditional independence test, we can use the same API to condition on another variable or set of
variables. First, let's generate a third variable $Z$ to condition on:

```
import numpy as np
  
rng = np.random.default_rng(0)
Z = rng.standard_normal((200, 1))
X = Z + rng.standard_normal(size=(200, 1))
Y = np.exp(Z + rng.standard_normal(size=(200, 1)))
```

Here, $X$ and $Y$ are dependent due to $Z$. Running an unconditional independence test yields:

```
from pywhy_stats import independence_test
 
result = independence_test(X, Y)
print("p-value:", result.pvalue, "Test statistic:", result.statistic)
```

Again, the p-value is very small, indicating a high likelihood that $X$ and $Y$ are dependent. Now,
let's condition on $Z$, which should render the variables as independent:

```
result = independence_test(X, Y, condition_on=Z)
print("p-value:", result.pvalue, "Test statistic:", result.statistic)
```

We observe that the p-value isn't small anymore. Indeed, if the variables were independent, we would expect the p-value
to be uniformly distributed on $[0, 1]$.

### (Conditional) k-sample test

In certain settings, you may be interested in testing the invariance between k (conditional) distributions. For example, say you have data collected over the same set of variables (X, Y) from humans ($P^1(X, Y)$) and bonobos ($P^2(X, Y)$). You can determine if the conditional distributions $P^1(Y | X) = P^2(Y | X)$ using conditional two-sample test.

First, we can create some simulated data that arise from two distinct distributions. However, the data generating Y is invariant across these two settings once we condition on X.

```
import numpy as np
  
rng = np.random.default_rng(0)
X1 = rng.standard_normal((200, 1))
X2 = rng.uniform(low=0.0, high=1.0, size=(200, 1))

Y1 = np.exp(X1 + rng.standard_normal(size=(200, 1)))
Y2 = np.exp(X2 + rng.standard_normal(size=(200, 1)))

groups = np.concatenate((np.zeros((200, 1)), np.ones((200, 1))))
X = np.concatenate((X1, X2))
Y = np.concatenate((Y1, Y2))
```

We test the hypothesis that $P^1(Y | X) = P^2(Y | X)$ now with the following code.

```
from pywhy_stats import conditional_ksample

# test that P^1(Y | X) = P^2(Y | X)
result = conditional_ksample.kcd.condind(X, Y, groups)

print("p-value:", result.pvalue, "Test statistic:", result.statistic)
```

## Dependencies

Minimally, pywhy-stats requires:

    * Python (>=3.8)
    * numpy
    * scipy
    * scikit-learn

## User Installation

If you already have a working installation of numpy and scipy, the easiest way to install pywhy-stats is using `pip`:

    pip install -U pywhy-stats

To install the package from github, clone the repository and then `cd` into the directory. You can then use `poetry` to install:

    poetry install

    # if you would like an editable install of pywhy-stats for dev purposes
    pip install -e .

# Contributing

We welcome contributions from the community. Please refer to our [contributing document](./CONTRIBUTING.md) and [developer document](./DEVELOPING.md) for information on developer workflows.