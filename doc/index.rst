**pywhy-stats**
===================

pywhy-stats is a Python package for representing causal graphs. For example, Acyclic
Directed Mixed Graphs (ADMG), also known as causal DAGs and Partial Ancestral Graphs (PAGs).
We build on top of ``networkx's`` ``MixedEdgeGraph`` such that we maintain all the well-tested and efficient
algorithms and data structures of ``networkx``.

We encourage you to use the package for your causal inference research and also build on top
with relevant Pull Requests.

See our examples for walk-throughs of how to use the package, or 

Please refer to our :ref:`user_guide` for details on all the tools that we
provide. You can also find an exhaustive list of the public API in the
:ref:`api_ref`. You can also look at our numerous :ref:`examples <general_examples>`
that illustrate the use of ``pywhy-stats`` in many different contexts.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting started:

   installation
   Reference API<api>
   Usage<use>
   User Guide<user_guide>
   tutorials/index
   whats_new

.. toctree::
   :hidden:
   :caption: Development

   License <https://raw.githubusercontent.com/py-why/pywhy-stats/main/LICENSE>
   Contributing <https://github.com/py-why/pywhy-stats/main/CONTRIBUTING.md>

Team
----

**pywhy-stats** is developed and maintained by pywhy.
To learn more about who specifically contributed to this codebase, see
`our contributors <https://github.com/py-why/pywhy-stats/graphs/contributors>`_ page.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
