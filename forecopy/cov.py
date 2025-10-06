"""
This module provides **functions for approximating forecast error covariance
matrices**, with support for both cross-sectional and temporal frameworks.
These covariance approximations are a key component of optimal combination
forecast reconciliation methods, since the choice of covariance structure
determines the weighting in least-squares adjustments.

Two main toolsets are included:

- :func:`cscov() <forecopy.cov.cscov>`: 
  For hierarchically, grouped, or otherwise linearly constrained series
  observed at the same frequency.

- :func:`tecov() <forecopy.cov.tecov>`: 
  For temporal hierarchies, where the same time series can be aggregated or
  linearly combined at multiple frequencies (e.g., monthly, quarterly,
  yearly).
"""
import jax.numpy as jnp
from forecopy.tools import cstools, tetools
from forecopy.fun import shrink_estim, sample_estim, factors, vec2hmat, is_PD

class cscov:
    """
    Cross-sectional covariances
    ---------------------------
    Approximate the cross-sectional base forecast error covariance matrix 
    under several reconciliation assumptions (Wickramasuriya et al., 2019; 
    Di Fonzo & Girolimetto, 2023).

    Parameters
    ----------

    ``params``: cstools
         A :class:`cstools <forecopy.tools.cstools>` object.

    ``res``: ndarray
        An :math:`(N \\times n)` optional numeric matrix containing the 
        residuals. This matrix is used to compute some covariance matrices.

    ``cov_mat``: jnp.ndarray
        If given, this :math:`(n \\times n)` matrix is returned by ``fit()`` 
        regardless of `comb`. Use this to plug in a fully custom covariance.

    Attributes
    ----------

    ``params``: cstools
        Cached params.

    ``n``: int
        Number of series.

    ``res``: jnp.ndarray
        Cached residual matrix.

    ``cov_mat``: jnp.ndarray
        Custom covariance to return as-is.
    
    References
    ----------

    * Di Fonzo, T. and Girolimetto, D. (2023), Cross-temporal forecast 
      reconciliation: Optimal combination method and heuristic alternatives, 
      `International Journal of Forecasting`, 39, 1, 39-57. 
      `DOI:10.1016/j.ijforecast.2021.08.004 <https://doi.org/10.1016/j.ijforecast.2021.08.004>`_
    * Wickramasuriya, S.L., Athanasopoulos, G. and Hyndman, R.J. (2019), 
      Optimal forecast reconciliation for hierarchical and grouped time series 
      through trace minimization, `Journal of the American Statistical 
      Association`, 114, 526, 804-819.
      `DOI:10.1080/01621459.2018.1448825 <https://doi.org/10.1080/01621459.2018.1448825>`_
    
    See Also
    --------
    :func:`shrink_estim <forecopy.fun.shrink_estim>`
    :func:`cstools <forecopy.tools.cstools>`
    :func:`csrec <forecopy.lsrec.csrec>`
    """
    def __init__(self, params: cstools = None, res: jnp.ndarray = None):
        if res is not None:
            assert res.shape[1] == self.n, "The size of the residuals does not match."
        self.params = params
        self.n = params.dim[0]
        self.res = res
        self.cov_mat = None

    def fit(self, comb: str | jnp.ndarray = 'ols', return_vector = False, shr_fun = shrink_estim, 
            mse = True):
        """
        Estimate a cross-sectional covariance approximation.

        Parameters
        ----------

        ``comb``: str or ndarray, default 'ols'
            The approximation/reconciliation assumption to use:

              - `ols`: identity error covariance matrix.
              - `str`: structural variances.
              - `wls`: series-wise variances from residuals `res`.
              - `shr`: shrunk covariance of `res` (Wickramasuriya et al., 2019).
              - `sam` : sample covariance of `res`.
              - A custom :math:`(n \\times n)` covariance matrix.

        ``return_vector``: bool, default `False`
            If True, return the diagonal of the matrix, only for 
            {'ols','str', 'wls'}. Otherwise return an :math:`(n \\times n)`
            matrix.

        ``shr_fun``: callable, default :func:`shrink_estim() <forecopy.fun.shrink_estim>`
            Shrinkage estimator used when `comb='shr'`. It must accept residuals
            `res` and `mse` and return a mapping with at least the key 'cov'
            containing the shrunk covariance :math:`(n \\times n)`.

        ``mse``: bool, default True
            When `True`, residual moments are not mean-corrected (i.e., MSE rather
            than unbiased variance). When `False`, apply mean correction.

        Returns
        -------
        A :math:`(n \\times n)` symmetric positive (semi-)definite matrix.
        """
        if not isinstance(comb, str):
            self.cov_mat = jnp.array(comb)
            assert self.cov_mat.shape[0] == self.cov_mat.shape[1] == self.n, "custom matrix should be square."
            assert is_PD(self.cov_mat), "custom matrix should be positive definite."
            return self.cov_mat
        elif comb == "ols":
            self.cov_mat = _cscov_ols(
                n = self.n, 
                return_vector = return_vector
                )
        elif comb == "str":
            self.cov_mat = _cscov_str(
                strc_mat = self.params.strc_mat(), 
                return_vector = return_vector
                )
        elif comb == "wls":
            self.cov_mat =_cscov_wls(
                res = self.res, 
                mse = mse, 
                return_vector = return_vector
                )
        elif comb == "shr":
            out = _cscov_shr(
                res = self.res, 
                mse = mse, 
                shr_fun = shr_fun
                )
            self.lmb = out.get('lambda')
            self.cov_mat = out.get('cov')
            return out.get('cov')
        elif comb == "sam":
            self.cov_mat =_cscov_sam(res = self.res, mse = mse)
        else:
            raise Exception("Error cscov")
        return self.cov_mat

class tecov:
    """
    Temporal covariances
    ---------------------------
    Approximate the temporal base forecast error covariance matrix 
    under several reconciliation assumptions (Di Fonzo & Girolimetto, 2023).

    Parameters
    ----------

    ``params``: tetools
         A :class:`tetools <forecopy.tools.tetools>` object.

    ``res``: ndarray
        A :math:`[N(k^\\ast+m) \\times 1]` optional numeric vector containing
        the residuals ordered from the lowest frequency to the highest
        frequency. This vector is used to compute some covariance matrices.

    ``cov_mat``: jnp.ndarray
        If given, this :math:`[(k^\\ast+m) \\times (k^\\ast+m)]` matrix is 
        returned by ``fit()`` regardless of `comb`.
        Use this to plug in a fully custom covariance.

    Attributes
    ----------

    ``params``: cstools
        Cached params.

    ``kt``: int
        Total sum of factors.

    ``res``: jnp.ndarray
        Cached residual matrix.

    ``cov_mat``: jnp.ndarray
        Custom covariance to return as-is.

    References
    ----------

    * Di Fonzo, T. and Girolimetto, D. (2023), Cross-temporal forecast 
      reconciliation: Optimal combination method and heuristic alternatives, 
      `International Journal of Forecasting`, 39, 1, 39-57. 
      `DOI:10.1016/j.ijforecast.2021.08.004 <https://doi.org/10.1016/j.ijforecast.2021.08.004>`_
    
    See Also
    --------
    :func:`shrink_estim <forecopy.fun.shrink_estim>`
    :func:`tetools <forecopy.tools.tetools>`
    :func:`terec <forecopy.lsrec.terec>`
    """
    def __init__(self, params: tetools = None, res: jnp.ndarray = None, cov_mat = None):
        self.params = params
        self.kt = params.kt
        self.res = res
        self.cov_mat = cov_mat

    def fit(self, comb = 'ols', return_vector = False, shr_fun = shrink_estim, mse = True):
        """
        Estimate a cross-sectional covariance approximation.

        Parameters
        ----------
        ``comb``: str, default 'ols'
            The approximation/reconciliation assumption to use:

              - `ols`: identity error covariance matrix.
              - `str`: structural variances.
              - `wlsv`: series-wise variances from residuals `res`.
              - `shr`: shrunk covariance of `res`.
              - `sam` : sample covariance of `res`.

        ``return_vector``: bool, default `False`
            If True, return the diagonal of the matrix, only for 
            {'ols','str','wlsv'}. Otherwise return an 
            :math:`[(k^\\ast+m) \\times (k^\\ast+m)]` matrix.

        ``shr_fun``: callable, default :func:`shrink_estim() <forecopy.fun.shrink_estim>`
            Shrinkage estimator used when `comb='shr'`. It must accept residuals
            `res` and `mse` and return a mapping with at least the key 'cov'
            containing the shrunk covariance 
            :math:`[(k^\\ast+m) \\times (k^\\ast+m)]`.

        ``mse``: bool, default True
            When `True`, residual moments are not mean-corrected (i.e., MSE 
            rather than unbiased variance). When `False`, apply mean correction.

        Returns
        -------
        A :math:`[(k^\\ast+m) \\times (k^\\ast+m)]` symmetric positive 
        (semi-)definite matrix.
        """
        if self.cov_mat is not None:
            return self.cov_mat
        elif comb == "ols":
            return _cscov_ols(
                n = self.kt, 
                return_vector = return_vector
                )
        elif comb == "str":
            return _cscov_str(
                strc_mat = self.params.strc_mat(), 
                return_vector = return_vector
                )
        elif comb == "wlsv":
            return _tecov_wlsv(
                agg_order = self.params.kset, 
                res = self.res,
                mse = mse, 
                return_vector = return_vector
                )
        elif comb == "shr":
            out = _tecov_shr(
                agg_order = self.params.kset, 
                res = self.res, 
                mse = mse, 
                shr_fun = shr_fun
                )
            self.lmb = out.get('lambda')
            return out.get('cov')
        elif comb == "sam":
            return _tecov_sam(
                agg_order = self.params.kset, 
                res = self.res,
                mse = mse
                )
        else:
            raise Exception("Error tecov")


def _cscov_ols(n, return_vector = False):
    if n is None:
        raise TypeError("Missing required argument: 'n'")

    if return_vector:
        return jnp.ones(n)
    else:
        return jnp.eye(n)

def _cscov_str(strc_mat, return_vector = False):
    if strc_mat is None:
        raise TypeError("Missing required argument: 'strc_mat'")
    
    if return_vector:
        return strc_mat.sum(1)
    else:
        return jnp.diag(strc_mat.sum(1))

def _cscov_wls(res, mse = True, return_vector = False):
    if res is None:
        raise TypeError("Missing required argument: 'res'")
    
    if not mse:
        res -= jnp.nanmean(res, 0)

    var = jnp.nanmean((res**2), 0)

    if return_vector:
        return var
    else:
        return jnp.diag(var)


def _cscov_shr(res, mse=True, shr_fun = shrink_estim):
    if res is None:
        raise TypeError("Missing required argument: 'res'")

    cov = shr_fun(x=res, mse=mse)

    return cov

def _cscov_sam(res, mse=True):
    if res is None:
        raise TypeError("Missing required argument: 'res'")

    return sample_estim(x=res, mse=mse)

def _tecov_wlsv(agg_order, res, mse: bool = True, return_vector = False):
    if agg_order is None:
        raise ValueError("Missing required argument: 'agg_order'")
    if res is None:
        raise ValueError("Missing required argument: 'res'")
    
    if isinstance(agg_order, list):
        kset = [int(x) for x in agg_order]
        kset = sorted(kset, reverse=True)
    else:
        kset = factors(int(agg_order))
    
    if len(res.shape) != 1:
        if res.shape[0] != 1:
            raise ValueError("'res' is not a vector.")
        res = res[0,:]

    m = max(kset)
    div = [int(m/k) for k in kset]
    N = int(res.shape[0]/sum([m/k for k in kset]))

    idk = jnp.repeat(jnp.array(kset), jnp.array(div)*N)
    var_freq = [sample_estim(x=res[idk==k], mse=mse).tolist() for k in kset]
    out = jnp.repeat(jnp.array(var_freq), jnp.array(div))

    if return_vector:
        return out
    else:
        return jnp.diag(out)


def _tecov_shr(agg_order, res, mse: bool = True, shr_fun = shrink_estim):
    if agg_order is None:
        raise ValueError("Missing required argument: 'agg_order'")
    if res is None:
        raise ValueError("Missing required argument: 'res'")
    
    if isinstance(agg_order, list):
        kset = [int(x) for x in agg_order]
        kset = sorted(kset, reverse=True)
    else:
        kset = factors(int(agg_order))
    
    if len(res.shape) != 1:
        if res.shape[0] != 1:
            raise ValueError("'res' is not a vector.")
        res = res[0,:]

    m = max(kset)
    N = res.shape[0]/sum([m/k for k in kset])
    res_mat = vec2hmat(vec=res, h=int(N), kset=kset)
    return shr_fun(x=res_mat, mse=mse)

def _tecov_sam(agg_order, res, mse: bool = True):
    if agg_order is None:
        raise ValueError("Missing required argument: 'agg_order'")
    if res is None:
        raise ValueError("Missing required argument: 'res'")
    
    if isinstance(agg_order, list):
        kset = [int(x) for x in agg_order]
        kset = sorted(kset, reverse=True)
    else:
        kset = factors(int(agg_order))
    
    if len(res.shape) != 1:
        if res.shape[0] != 1:
            raise ValueError("'res' is not a vector.")
        res = res[0,:]

    m = max(kset)
    N = res.shape[0]/sum([m/k for k in kset])
    res_mat = vec2hmat(vec=res, h=int(N), kset=kset)
    return sample_estim(x=res_mat, mse=mse)