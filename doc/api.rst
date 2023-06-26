.. _api_ref:

###
API
###

:py:mod:`pywhy_stats`:

.. automodule:: pywhy_stats
   :no-members:
   :no-inherited-members:

This is the application programming interface (API) reference
for classes (``CamelCase`` names) and functions
(``underscore_case`` names) of pywhy-stats, grouped thematically by analysis
stage.

********************
Independence Testing
********************
Pyhy-Stats experimentally provides an interface for conditional independence
testing and conditional discrepancy testing (also known as k-sample conditional
independence testing).

High-level Independence Testing
===============================

The easiest way to run a (conditional) independence test is to use the
:py:func:`independence_test` function. This function takes inputs and
will try to automatically pick the appropriate test based on the input.

Note: this is only meant for beginnners, and the result should be interpreted
with caution as the ability to choose the optimal test is limited. When
one uses the wrong test for the type of data and assumptions they have,
then typically you will get less statistical power.

.. currentmodule:: pywhy_stats
.. autosummary::
   :toctree: generated/

   independence_test
   Methods


All independence tests return a ``PValueResult`` object, which
contains the p-value and the test statistic and optionally additional information.

.. currentmodule:: pywhy_stats.pvalue_result
.. autosummary::
   :toctree: generated/

   PValueResult

(Conditional) Independence Testing
==================================

Testing for conditional independence among variables is a core part
of many data analysis procedures.

.. currentmodule:: pywhy_stats.independence
.. autosummary::
   :toctree: generated/
   
   fisherz
   kci


(Conditional) Discrepancy Testing
=================================

Testing for invariances among conditional distributions is a core part
of many data analysis procedures.

.. currentmodule:: pywhy_stats.discrepancy
.. autosummary::
   :toctree: generated/
   
   bregman
   kcd

