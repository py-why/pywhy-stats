[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CircleCI](https://circleci.com/gh/py-why/pywhy-stats/tree/main.svg?style=svg)](https://circleci.com/gh/py-why/pywhy-stats/tree/main)
[![unit-tests](https://github.com/py-why/pywhy-stats/actions/workflows/main.yml/badge.svg)](https://github.com/py-why/pywhy-stats/actions/workflows/main.yml)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![codecov](https://codecov.io/gh/py-why/pywhy-stats/branch/main/graph/badge.svg?token=H1reh7Qwf4)](https://codecov.io/gh/py-why/pywhy-stats)

# PyWhy-Stats

Pywhy-stats serves as Python library for implementations of various statistical methods, such as (un)conditional independence tests, which can be utilized in tasks like causal discovery.

# Documentation

See the [development version documentation](https://py-why.github.io/pywhy-stats/dev/index.html).

Or see [stable version documentation](https://py-why.github.io/pywhy-stats/stable/index.html)

# Installation

Installation is best done via `pip` or `conda`. For developers, they can also install from source using `pip`. See [installation page](TBD) for full details.

## Dependencies

Minimally, pywhy-stats requires:

    * Python (>=3.8)
    * numpy
    * scipy
    * scikit-learn

## User Installation

If you already have a working installation of numpy and scipy, the easiest way to install pywhy-stats is using `pip`:

    # doesn't work until we make an official release :p
    pip install -U pywhy-stats

To install the package from github, clone the repository and then `cd` into the directory. You can then use `poetry` to install:

    poetry install

    # if you would like an editable install of pywhy-stats for dev purposes
    pip install -e .

# Contributing

We welcome contributions from the community. Please refer to our [contributing document](./CONTRIBUTING.md) and [developer document](./DEVELOPING.md) for information on developer workflows.
