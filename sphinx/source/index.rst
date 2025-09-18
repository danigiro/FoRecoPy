FoRecoPy: Forecast Reconciliation in Python
===========================================

Forecast Reconciliation is a a post-forecasting process aimed to 
improve the accuracy and align forecasts for a system of linearly constrained 
(e.g. hierarchical/grouped) time series. 

The ``FoRecoPy`` package is inspired by the **R** package `FoReco <https://danigiro.github.io/FoReco>`_ and brings similar functionality 
to **Python**. It is designed for researchers, practitioners, and data scientists 
who use Python for time series forecasting and want access to state-of-the-art 
reconciliation methods.

Currently, ``FoRecoPy`` supports:

* Regression-based reconciliation (e.g. minimum trace)
* Both cross-sectional reconciliation (hierarchical, grouped and linearly 
  constrained time series) and temporal reconciliation (multiple aggregation 
  frequencies)

Future versions will expand the scope to include the cross-temporal framework,
non-negative constraints and probabilistic reconciliation.

.. toctree::
   :caption: Table of contents
   :maxdepth: 1

   overview
   doc_lsrec
   doc_tools
   doc_cov
   doc_fun
   about
   changelog
   genindex
