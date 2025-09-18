Overview
========

``FoRecoPy`` is a Python package for **forecast reconciliation** of
**general linearly constrained multiple time series**. It provides methods
for adjusting base forecasts so that they satisfy known linear relationships,
while remaining as close as possible to the original forecasts in a
least-squares sense.

Linearly constrained structures arise in many contexts:

- **Hierarchical or grouped systems**  
  e.g., regional sales summing to national totals, or product-level demand
  aggregating to category totals.

- **General linear constraints**  
  e.g., balances in economic accounts, conservation laws in energy flows,
  or other domain-specific relationships.

- **Temporal aggregation systems**  
  e.g., monthly forecasts that should sum to quarterly or yearly values.

Naive forecasts often violate these constraints, producing incoherent results.
``FoRecoPy`` reconciles the forecasts by enforcing linear consistency across all
series, using optimal combination methods and a variety of covariance
approximation strategies.

It supports both **projection** and **structural** approaches,
offers multiple solvers for efficiency and scalability, and includes an option
to **enforce non-negativity** on reconciled forecasts. This makes ``FoRecoPy`` a
general-purpose environment for forecast reconciliation in any setting where
linear constraints must be respected.

Examples - Cross-sectional framework
------------------------------------

.. code-block:: python

    import numpy as np
    import forecopy as rpy

    np.random.seed(123)

    # Simulation parameters for base and residuals
    h = 2    # Forecast horizons
    N = 100  # Number of residuals
    agg_mat = np.array([1,1,1,1,1,1,0,0,0,0,1,1]).reshape(3,4) # Aggregation matrix
    bts_mean = np.repeat(5, agg_mat.shape[1]) # Bottom time series' mean
    mean = np.concatenate([agg_mat @ bts_mean, bts_mean]) # All time series' mean

    # Simulated base forecasts
    base = np.random.normal(
        loc = np.concatenate([mean for i in range(h)]),
        size = sum(agg_mat.shape)*h
        ).reshape([h,sum(agg_mat.shape)])

    # Simulated residuals 
    res = np.random.normal(
        loc = np.concatenate([mean for i in range(N)]),
        size = sum(agg_mat.shape)*N
        ).reshape([N,sum(agg_mat.shape)]) - mean

    # Optimal forecast reconciliation with shrunk covariance - input: aggregation matrix
    reco_agg = rpy.csrec(base=base, agg_mat=agg_mat, comb='shr', res = res)
    
    # Optimal forecast reconciliation with shrunk covariance - input: constraints matrix
    cons_mat = rpy.cstools(agg_mat=agg_mat).cons_mat()
    reco_cons = rpy.csrec(base=base, cons_mat=cons_mat, comb='shr', res = res)
    # np.all(reco_cons==reco_agg).tolist()

    # Non-negative solution 
    mean_neg = mean
    mean_neg[agg_mat.shape[0]+1] = -10

    # Simulated negative base forecasts
    base_neg = np.random.normal(
        loc = np.concatenate([mean_neg for i in range(h)]),
        size = sum(agg_mat.shape)*h
        ).reshape([h,sum(agg_mat.shape)])

    # Optimal forecast reconciliation with negative values
    reco_neg = rpy.csrec(base=base_neg, agg_mat=agg_mat, comb='shr', res = res)

    # Optimal forecast reconciliation with non-negative values
    reco_sntz = rpy.csrec(base=base_neg, agg_mat=agg_mat, comb='shr', res = res, nn = True)


Examples - Temporal framework
-----------------------------

.. code-block:: python

    import numpy as np
    import forecopy as rpy

    np.random.seed(123)

    # Simulation parameters for base and residuals
    h = 2    # Forecast horizons for the lowest frequency time series
    N = 100  # Number of residuals for the lowest frequency time series
    agg_order = 12 # Max. aggregation order
    strc_mat_te = rpy.tetools(agg_order=agg_order).strc_mat()
    mean = np.array(strc_mat_te @ np.repeat(3, strc_mat_te.shape[1]))

    # Simulated base forecasts
    base = np.random.normal(
        loc = np.repeat(mean, h),
        size = strc_mat_te.shape[0]*h
        )

    # Simulated residuals 
    res = np.random.normal(
        loc = np.repeat(mean, N),
        size = strc_mat_te.shape[0]*N
        ) - np.repeat(mean, N)

    # Optimal forecast reconciliation with series-wise variances
    reco = rpy.terec(base=base, agg_order=agg_order, comb='wlsv', res = res)

    # Non-negative solution 
    mean_neg = mean
    mean_neg[-np.diff(strc_mat_te.shape)+1] = -30

    # Simulated negative base forecasts
    base_neg = np.random.normal(
        loc = np.repeat(mean_neg, h),
        size = strc_mat_te.shape[0]*h
        )

    # Optimal forecast reconciliation with negative values
    reco_neg = rpy.terec(base=base_neg, agg_order=agg_order, comb='wlsv', res = res)

    # Optimal forecast reconciliation with non-negative values
    reco_sntz = rpy.terec(base=base_neg, agg_order=agg_order, comb='wlsv', res = res, nn = True)
