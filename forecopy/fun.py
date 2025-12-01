import pandas as pd
import jax.numpy as jnp
import scipy.sparse as sps
import jax as jax
from itertools import chain
import lineax as lx
import jax.scipy.linalg as jsl

def factors(n):
    kset = list(chain.from_iterable([[i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0]))
    kset = list(set(kset))
    kset.sort(reverse=True)
    return kset

def vec2hmat(vec, h, kset):
    if len(vec.shape) == 2:
        vec = vec[0,:]

    m = max(kset)
    freq = m/jnp.array(kset)
    idx = jnp.tile(jnp.array([range(1, h+1)]), len(kset))[0, :]
    idx = jnp.repeat(idx, jnp.repeat(freq, h).astype(int))
    idx_sort = jnp.argsort(idx, stable=True)

    return vec[idx_sort].reshape(h, int(len(idx)/h))

def hmat2vec(hmat, kset):
    h = hmat.shape[0]
    m = max(kset)
    freq = m/jnp.array(kset)
    it = jnp.tile(jnp.repeat(freq, freq.astype(int)), h)

    return (hmat.T.flatten('F'))[jnp.argsort(it, stable=True)]

def FoReco2pd(reco: jnp.ndarray, type: str = "cs", labels: tuple = None):
    if labels is None:
        labels = [f"s-{i+1}" for i in range(reco.shape[1])]

    index = [f"h-{i+1}" for i in range(reco.shape[0])]

    df = pd.DataFrame(data=reco, index=index, columns=labels)

    return df

def isDiag(M):
    if len(M.shape) == 1:
        return True
    elif sps.issparse(M):
        i, j, _= sps.find(M)
        return jnp.all(i == j)
    else:
        i, j = M.shape
        assert i == j 
        test = M.reshape(-1)[:-1].reshape(i-1, j+1)
        return ~jnp.any(test[:, 1:])

def lin_sys(lhs, rhs, solver = 'default'):
    if solver == 'lineax':
        lhs = lx.MatrixLinearOperator(lhs, lx.positive_semidefinite_tag)
        solver = lx.Cholesky()
        state = solver.init(lhs, options={})
        def fun_solver(x):
            sol = solver.compute(state, x, options={})
            return sol[0]
        vmap_solver = jax.vmap(fun_solver, 1)
        return vmap_solver(rhs).T
    else:
        return jsl.solve(lhs, rhs)

def _mcrossprod(x):
    return jnp.dot(x.T,x)

def _covcor(cov):
    d = jnp.sqrt(jnp.diag(cov))
    corm = ((cov / d).T)/d
    corm = jnp.fill_diagonal(corm, 1.0, inplace = False)
    return corm

def sample_estim(x, mse=True):
    if jnp.any(jnp.isnan(x)):
        n = (~jnp.isnan(x)).sum(0)
        n = n * jnp.ones(len(n))[:, None]
        if not mse:
            x -= jnp.nanmean(x, 0)
        x[jnp.isnan(x)] = 0
        return (x.T @ x) * (1/jnp.minimum(n,n.T))
    else:
        if not mse:
            return jnp.cov(x.T)
        else:
            return jnp.dot(x.T, x) / x.shape[0]

def shrink_estim(x, mse=True):
    """
    Shrinkage of the covariance matrix
    ----------------------------------
    Shrinkage of the covariance matrix according to Schafer and Strimmer (2005).

    Parameters
    ----------

    ``x``: ndarray
        A numeric matrix containing the residuals.

    ``mse``: bool, default True
        When `True`, residual moments are not mean-corrected (i.e., MSE rather
        than unbiased variance). When `False`, apply mean correction.

    Returns
    -------
    A shrunk covariance matrix.

    References
    ----------

    * Schafer, J.L. and Strimmer, K. (2005), A shrinkage approach to large-scale
      covariance matrix estimation and implications for functional genomics,
      `Statistical Applications in Genetics and Molecular Biology`, 4, 1

    See Also
    --------
    :func:`tecov <forecopy.cov.tecov>`
    :func:`cscov <forecopy.cov.cscov>`
    """
    n = x.shape[0]

    covm = sample_estim(x, mse = mse)
    diag_covm = jnp.diag(covm)

    if n>3:
        xs = x/jnp.sqrt(diag_covm)

        vS = (1 / (n * (n - 1))) * (_mcrossprod(xs**2) - ((1 / n) * (_mcrossprod(xs)**2)))
        vS = jnp.fill_diagonal(vS, 0.0, inplace = False)
        corm = _covcor(covm)
        corm = jnp.fill_diagonal(corm, 0.0, inplace = False)
        corm = corm**2
        lam = (jnp.sum(vS)/jnp.sum(corm)).item(0)
        lam = max(min(lam, 1), 0)
    else:
        lam = 1
    
    shrink_cov = lam * jnp.diag(diag_covm) + (1 - lam) * covm
    return {'cov': shrink_cov, 'lambda': lam}

def is_PD(matrix: jnp.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if a matrix is positive definite.

    Parameters
    ----------
    matrix : jnp.ndarray
        The matrix to validate.
    tol : float, default 1e-10
        Tolerance for symmetry check.

    Returns
    -------
    bool
        True if the matrix is a valid covariance matrix.
    """
    # Check if matrix is square
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return False

    # Check if matrix is symmetric
    if jnp.max(jnp.abs(matrix - matrix.T)) > tol:
        return False

    # Check if matrix is positive definite using Cholesky decomposition
    try:
        jnp.linalg.cholesky(matrix)
        return True
    except jax.linalg.LinAlgError:
        return False
