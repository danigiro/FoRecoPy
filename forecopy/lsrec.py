"""
This module provides **functions for optimal (least-squares) forecast
combination**, with support for both cross-sectional and temporal
frameworks.

Two main reconciliation functions are included:

- :func:`csres() <forecopy.lsrec.csrec>`: 
  For hierarchically, grouped, or otherwise linearly constrained series
  observed at the same frequency.

- :func:`teres() <forecopy.lsrec.terec>`: 
  For temporal hierarchies, where the same time series can be aggregated or
  linearly combined at multiple frequencies (e.g., monthly, quarterly,
  yearly).
"""
import jax.numpy as jnp
from typing import Optional

# From forecopy
from forecopy.tools import cstools, tetools
from forecopy.cov import cscov, tecov
from forecopy.reco import _reconcile
from forecopy.fun import vec2hmat, hmat2vec

def csrec(
    base: jnp.ndarray,
    agg_mat: Optional[jnp.ndarray] = None,
    cons_mat: Optional[jnp.ndarray] = None,
    comb: str | jnp.ndarray = 'ols',
    res: Optional[jnp.ndarray] = None,
    approach: str = 'proj',
    solver: str = 'default',
    tol: float = 1e-6,
    nn: bool = False,
    **kwargs
):
    """
    Optimal combination cross-sectional reconciliation
    --------------------------------------------------
    
    This function performs optimal (in least squares sense) combination 
    cross-sectional forecast reconciliation for a linearly constrained 
    (e.g., hierarchical/grouped) multiple time series (Wickramasuriya et al., 
    2019, Panagiotelis et al., 2022, Girolimetto and Di Fonzo, 2023). The 
    reconciled forecasts are calculated using either a projection approach 
    (Byron, 1978, 1979) or the equivalent structural approach by Hyndman et 
    al. (2011).

    Parameters
    ----------

    ``base``: ndarray
        A :math:`(h \\times n)` numeric matrix containing the base forecasts 
        to be reconciled; :math:`h` is the forecast horizon, and :math:`n` 
        is the total number of time series (:math:`n = n_a + n_b`).

    ``agg_mat``: ndarray, default None
        A :math:`(n_a \\times n_b)` numeric matrix representing the
        cross-sectional aggregation matrix (alternative to ``cons_mat``).
        It maps the :math:`n_b` bottom-level (free) variables
        into the :math:`n_a` upper (constrained) variables.

    ``cons_mat``: ndarray, default None
        A :math:`(n_a \\times n)` numeric matrix representing the
        cross-sectional zero constraints (alternative to ``agg_mat``).

    ``comb``: str | ndarray, default `ols`
        A string specifying the reconciliation method. For a complete list, 
        see :func:`cscov() <forecopy.cov.cscov>`
        Alternatively, a custom :math:`(n \\times n)` covariance matrix.

    ``res``: ndarray, default None
        An :math:`(N \\times n)` optional numeric matrix containing the 
        residuals. This matrix is used to compute some covariance matrices.

    ``approach``: str, default `proj`
        A string specifying the approach used to compute the reconciled forecasts. 
        Options include: 

        * `proj`: Projection approach according to Byron (1978, 1979).
        * `strc`: Structural approach as proposed by Hyndman et al. (2011). 
        * `proj_tol`, `strc_tol`: implementation using the sparse matrix approach. 
          These methods cannot ensure identical results to the non-sparse version.
    
    ``solver``: str, default `default`
        Linear solvers: `default` and `linearx`.

    ``tol``: float
        Tolerance value for some linear solvers.

    ``nn``: bool
        If `True`, enforces non-negativity on reconciled forecasts using 
        the heuristic "set-negative-to-zero" (Di Fonzo and Girolimetto, 2023). 
        Default is `False`.

    ``**kwargs``: Arguments passed on to :func:`cscov() <forecopy.cov.cscov>`.

    Returns
    -------
    A :math:`(h \\times n)` numeric matrix of cross-sectional reconciled 
    forecasts.

    References
    ----------
    
    * Byron, R.P. (1978), The estimation of large social account matrices, 
      `Journal of the Royal Statistical Society, Series A`, 141, 3, 359-367. 
      `DOI:10.2307/2344807 <https://doi.org/10.2307/2344807>`_
    * Byron, R.P. (1979), Corrigenda: The estimation of large social account 
      matrices, `Journal of the Royal Statistical Society, Series A`, 142(3), 
      405. `DOI:10.2307/2982515 <https://doi.org/10.2307/2982515>`_
    * Di Fonzo, T. and Girolimetto, D. (2023), Spatio-temporal reconciliation 
      of solar forecasts, `Solar Energy`, 251, 13-29.
      `DOI:10.1016/j.solener.2023.01.003 <https://doi.org/10.1016/j.solener.2023.01.003>`_
    * Girolimetto, D. and Di Fonzo, T. (2023), Point and probabilistic forecast 
      reconciliation for general linearly constrained multiple time series,
      `Statistical Methods & Applications`, 33, 581-607. 
      `DOI:10.1007/s10260-023-00738-6 <https://doi.org/10.1007/s10260-023-00738-6>`_
    * Hyndman, R.J., Ahmed, R.A., Athanasopoulos, G. and Shang, H.L. (2011),
      Optimal combination forecasts for hierarchical time series,
      `Computational Statistics & Data Analysis`, 55, 9, 2579-2589.
      `DOI:10.1016/j.csda.2011.03.006 <https://doi.org/10.1016/j.csda.2011.03.006>`_
    * Panagiotelis, A., Athanasopoulos, G., Gamakumara, P. and Hyndman, R.J. 
      (2021), Forecast reconciliation: A geometric view with new insights on 
      bias correction, `International Journal of Forecasting`, 37, 1, 343-359.
      `DOI:10.1016/j.ijforecast.2020.06.004 <https://doi.org/10.1016/j.ijforecast.2020.06.004>`_
    * Wickramasuriya, S.L., Athanasopoulos, G. and Hyndman, R.J. (2019), 
      Optimal forecast reconciliation for hierarchical and grouped time series 
      through trace minimization, `Journal of the American Statistical 
      Association`, 114, 526, 804-819.
      `DOI:10.1080/01621459.2018.1448825 <https://doi.org/10.1080/01621459.2018.1448825>`_

    See Also
    --------
    :func:`cstools <forecopy.tools.cstools>`
    :func:`cscov <forecopy.cov.cscov>`
    """
    if len(base.shape) == 1:
        base = base[None, :]

    params = cstools(agg_mat=agg_mat, cons_mat=cons_mat)

    id_nn = None
    if params.agg_mat is not None:
        id_nn = [False for i in range(0,params.dim[1])] + [True for i in range(0,params.dim[2])]
    
    if base.shape[1] != params.dim[0]:
        raise ValueError("Incorrect base columns dimension.")

    cov_mat = cscov(
        params=params, 
        res=res
        ).fit(comb=comb, return_vector=True, **kwargs)


    reco = _reconcile(
        base=base, 
        cov_mat=cov_mat, 
        params=params,
        id_nn=id_nn
        )

    rf = reco.fit(approach=approach, solver=solver, tol=tol, nn=nn)
    return rf


def terec(
    base: jnp.ndarray,
    agg_order: list = None,
    fh: int = 1,
    comb: str = 'ols',
    res: jnp.ndarray = None,
    cov_mat: jnp.ndarray = None,
    tew: str = "sum",
    approach: str = 'proj',
    solver: str = 'default',
    tol: float = 1e-6, 
    nn: bool = False,
    **kwargs
):
    """
    Optimal combination temporal reconciliation
    --------------------------------------------------
    This function performs forecast reconciliation for a single time
    series using temporal hierarchies (Athanasopoulos et al., 2017). 
    The reconciled forecasts can be computed using either a projection 
    approach (Byron, 1978, 1979) or the equivalent structural 
    approach by Hyndman et al. (2011). 

    Parameters
    ----------

    ``base``: ndarray
        A :math:`[N(k^\\ast + m) \\times 1]` numeric vector containing the
        base forecasts to be reconciled, ordered from lowest to highest 
        frequency; :math:`m` is the maximum aggregation order, :math:`k^\\ast` 
        is the sum of a chosen subset of the :math:`p - 1` factors of :math:`m` 
        (excluding :math:`m` itself) and :math:`h` is the forecast horizon for 
        the lowest frequency time series.

    ``agg_order``: list
        Highest available sampling frequency per seasonal cycle (max. order 
        of temporal aggregation, :math:`m`), or a list representing a
        subset of :math:`p` factors of :math:`m`.

    ``fh``: int
        Forecast horizon for the lowest frequency (most temporally aggregated)
        series.

    ``comb``: str, default `ols`
        A string specifying the reconciliation method. For a complete list, see 
        :func:`tecov() <forecopy.cov.tecov>`

    ``res``: ndarray
        A :math:`[N(k^\\ast+m) \\times 1]` optional numeric vector containing
        the residuals ordered from the lowest frequency to the highest
        frequency. This vector is used to compute some covariance matrices.

    ``cov_mat``: jnp.ndarray
        An :math:`[(k^\\ast+m) \\times (k^\\ast+m)]` covariance matrix 
        (alternative to ``comb``).

    ``tew``: str, default `sum`
        Temporal aggregation weighting scheme applied within each high-to-low
        block when building the aggregation matrix:

        * `sum`: sum over the block.
        * `avg`: arithmetic average over the block.
        * `last`: take the last element in the block.
        * `first`: take the first element in the block.

    ``approach``: str, default `proj`
        A string specifying the approach used to compute the reconciled forecasts. 
        Options include: 

        * `proj`: Projection approach according to Byron (1978, 1979).
        * `strc`: Structural approach as proposed by Hyndman et al. (2011). 
        * `proj_tol`, `strc_tol`: implementation using the sparse matrix approach. 
          These methods cannot ensure identical results to the non-sparse version.
    
    ``solver``: str, default `default`
        Linear solvers: `default` and `linearx`.

    ``tol``: float
        Tolerance value for some linear solvers.

    ``nn``: bool
        If `True`, enforces non-negativity on reconciled forecasts using 
        the heuristic "set-negative-to-zero" (Di Fonzo and Girolimetto, 2023). 
        Default is `False`.

    ``**kwargs``: Arguments passed on to :func:`tecov() <forecopy.cov.tecov>`.

    Returns
    -------
    A :math:`[h(k^\\ast+m) \\times 1]` numeric vector of temporal reconciled 
    forecasts.

    References
    ----------
    
    * Athanasopoulos, G., Hyndman, R.J., Kourentzes, N. and Petropoulos, F. 
      (2017), Forecasting with Temporal Hierarchies, `European Journal of 
      Operational Research`, 262, 1, 60-74. 
      `DOI:10.1016/j.ejor.2017.02.046 <https://doi.org/10.1016/j.ejor.2017.02.046>`_
    * Byron, R.P. (1978), The estimation of large social account matrices, 
      `Journal of the Royal Statistical Society, Series A`, 141, 3, 359-367. 
      `DOI:10.2307/2344807 <https://doi.org/10.2307/2344807>`_
    * Byron, R.P. (1979), Corrigenda: The estimation of large social account matrices,
      `Journal of the Royal Statistical Society, Series A`, 142(3), 405.
      `DOI:10.2307/2982515 <https://doi.org/10.2307/2982515>`_
    * Di Fonzo, T. and Girolimetto, D. (2023), Spatio-temporal reconciliation 
      of solar forecasts, `Solar Energy`, 251, 13-29.
      `DOI:10.1016/j.solener.2023.01.003 <https://doi.org/10.1016/j.solener.2023.01.003>`_
    * Hyndman, R.J., Ahmed, R.A., Athanasopoulos, G. and Shang, H.L. (2011),
      Optimal combination forecasts for hierarchical time series,
      `Computational Statistics & Data Analysis`, 55, 9, 2579-2589.
      `DOI:10.1016/j.csda.2011.03.006 <https://doi.org/10.1016/j.csda.2011.03.006>`_

    See Also
    --------
    :func:`tetools <forecopy.tools.tetools>`
    :func:`tecov <forecopy.cov.tecov>`
    """
    if len(base.shape) != 1:
        if base.shape[0] != 1:
            raise ValueError("Base is not a vector.")
        base = base[0,:]
        
    params = tetools(agg_order=agg_order, tew=tew, fh=fh)
    
    id_nn = None
    if params._agg_mat is not None:
        id_nn = [False for i in range(0,params.ks)] + [True for i in range(0,params.m)]
    
    if base.shape[0] % params.kt != 0:
        raise ValueError("Incorrect base length.")
    else:
        h = int(base.shape[0] / params.kt)
    
    base = vec2hmat(vec = base, h = h, kset = params.kset)

    cov_mat = tecov(
        params=params, 
        res=res, 
        cov_mat=cov_mat
        ).fit(comb=comb, return_vector=True, **kwargs)

    if cov_mat.shape[0] != params.kt:
        raise ValueError("Incorrect covariance dimensions. Check 'res' columns dimension.")
    
    reco = _reconcile(
        base=base, 
        cov_mat=cov_mat, 
        params=params,
        id_nn=id_nn
        )

    rf = reco.fit(approach=approach, solver=solver, tol=tol, nn=nn)
    rf = hmat2vec(hmat = rf, kset = params.kset)
    return rf