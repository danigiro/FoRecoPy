import numpy as np
import jax as jax
import jax.numpy as jnp
import scipy.sparse as sps
from functools import partial
from forecopy.fun import isDiag, lin_sys


class _reconcile:
    def __init__(
        self,
        base: jnp.ndarray = None,
        cov_mat: jnp.ndarray = None,
        params=None,
        id_nn: list = None,
    ):
        self.base = base
        self.params = params
        if isDiag(cov_mat) and len(cov_mat.shape) != 1:
            self.cov_mat = cov_mat.diagonal()
        else:
            self.cov_mat = cov_mat
        self.id_nn = id_nn

    def fit(
        self, approach="proj", solver="default", tol=1e-12, nn=False, immutable=None
    ):
        if immutable is not None:
            if approach != "proj":
                raise ValueError(
                    "The 'immutable' option is only available with the 'proj' approach."
                )
            else:
                approach = "proj_immutable"

        if approach == "proj":
            reco = rproj(
                base=self.base,
                cons_mat=self.params.cons_mat(),
                cov_mat=self.cov_mat,
                solver=solver,
            )
        elif approach == "proj_tol":
            reco = rproj_tol(
                base=self.base,
                cons_mat=self.params.cons_mat(),
                cov_mat=self.cov_mat,
                tol=tol,
            )
        elif approach == "strc":
            reco = rstrc(
                base=self.base,
                strc_mat=self.params.strc_mat(),
                cov_mat=self.cov_mat,
                solver=solver,
            )
        elif approach == "strc_tol":
            reco = rstrc_tol(
                base=self.base,
                strc_mat=self.params.strc_mat(),
                cov_mat=self.cov_mat,
                solver=solver,
                tol=tol,
            )
        elif approach == "proj_immutable":
            reco = rproj_immutable(
                base=self.base,
                cons_mat=self.params.cons_mat(),
                cov_mat=self.cov_mat,
                immutable=immutable,
                solver=solver,
            )
        else:
            raise ValueError(f"The '{approach}' approach is not available.")

        if nn:
            reco = sntz(reco=reco, strc_mat=self.params.strc_mat(), id_nn=self.id_nn)
        return reco


@partial(jax.jit, static_argnames=["solver"])
def rstrc(base, strc_mat, cov_mat, solver="default"):
    if strc_mat is None:
        raise TypeError("Missing required argument: 'strc_mat'")

    if strc_mat.shape[0] != cov_mat.shape[0] or base.shape[1] != cov_mat.shape[0]:
        raise ValueError("The size of the matrices does not match.")

    if len(cov_mat.shape) == 1:
        strc_cov = strc_mat.T * jnp.reciprocal(cov_mat)
    else:
        strc_cov = lin_sys(lhs=cov_mat, rhs=strc_mat, solver=solver).T

    lhs = strc_cov @ strc_mat
    rhs = strc_cov @ base.T
    lm = lin_sys(lhs=lhs, rhs=rhs, solver=solver)
    out = (strc_mat @ lm).T
    return out


def rstrc_tol(base, strc_mat, cov_mat, solver="default", tol=1e-5):
    if strc_mat is None:
        raise TypeError("Missing required argument: 'strc_mat'")

    if strc_mat.shape[0] != cov_mat.shape[0] or base.shape[1] != cov_mat.shape[0]:
        raise ValueError("The size of the matrices does not match.")

    if len(cov_mat.shape) == 1:
        strc_mat = sps.csr_matrix(strc_mat)
        strc_cov = (strc_mat.T).multiply(np.reciprocal(cov_mat))
    else:
        strc_cov = lin_sys(lhs=cov_mat, rhs=strc_mat, solver=solver).T
        strc_cov = sps.csr_matrix(strc_cov)

    def matvec_action(y):
        b = strc_cov @ base.T @ y
        A = sps.linalg.LinearOperator(
            (b.size, b.size), matvec=lambda v: strc_cov @ (strc_mat @ v)
        )
        btilde, _ = sps.linalg.bicgstab(A, b, atol=tol)
        return btilde

    bts = sps.linalg.LinearOperator(
        (strc_mat.shape[1], base.shape[0]), matvec=matvec_action
    )
    out = (strc_mat @ (bts @ np.identity(bts.shape[1]))).T
    return out


def rproj_tol(base, cons_mat, cov_mat, tol=1e-12):
    if cons_mat is None:
        raise TypeError("Missing required argument: 'cons_mat'")

    if cons_mat.shape[1] != cov_mat.shape[0] or base.shape[1] != cov_mat.shape[0]:
        raise ValueError("The size of the matrices does not match.")

    if len(cov_mat.shape) == 1:
        cons_mat = sps.csr_matrix(cons_mat)
        cons_cov = (cons_mat).multiply(cov_mat)
    else:
        cons_cov = sps.csr_matrix(cov_mat)
        cons_cov = cons_mat @ cov_mat

    def matvec_action(y):
        b = cons_mat @ base.T @ y
        A = sps.linalg.LinearOperator(
            (b.size, b.size), matvec=lambda v: cons_cov @ (cons_mat.T @ v)
        )
        btilde, _ = sps.linalg.bicgstab(A, b, atol=tol)
        return btilde

    lm = sps.linalg.LinearOperator(
        (cons_mat.shape[0], base.shape[0]), matvec=matvec_action
    )
    lm = lm @ np.identity(lm.shape[1])
    out = base - (cons_cov.T @ lm).T
    return out


@partial(jax.jit, static_argnames=["solver"])
def rproj(base, cons_mat, cov_mat, solver="default"):
    if cons_mat is None:
        raise TypeError("Missing required argument: 'cons_mat'")

    if cons_mat.shape[1] != cov_mat.shape[0] or base.shape[1] != cov_mat.shape[0]:
        raise ValueError("The size of the matrices does not match.")

    if len(cov_mat.shape) == 1:
        cov_cons = (cov_mat * cons_mat).T
    else:
        cov_cons = cov_mat @ cons_mat.T

    lhs = cons_mat @ cov_cons
    rhs = cons_mat @ base.T
    lm = lin_sys(lhs=lhs, rhs=rhs, solver=solver)
    out = base - (cov_cons @ lm).T
    return out


def sntz(reco, strc_mat, id_nn):
    if strc_mat is None:
        raise TypeError("Missing required argument: 'strc_mat'")
    reco = reco[:, id_nn]
    # reco[reco<0] = 0
    reco = reco.at[jnp.where(reco < 0)].set(0)
    return reco @ strc_mat.T


@partial(jax.jit, static_argnames=["solver"])
def rproj_immutable(base, cons_mat, cov_mat, immutable, solver="default"):
    if cons_mat is None:
        raise TypeError("Missing required argument: 'cons_mat'")

    if cons_mat.shape[1] != cov_mat.shape[0] or base.shape[1] != cov_mat.shape[0]:
        raise ValueError("The size of the matrices does not match.")

    imm_cons_mat = jax.nn.one_hot(immutable, base.shape[1])
    compl_cons_mat = jnp.vstack([cons_mat, imm_cons_mat])

    # check immutable feasibility
    # TODO

    if len(cov_mat.shape) == 1:
        cov_cons = (cov_mat * compl_cons_mat).T
    else:
        cov_cons = cov_mat @ compl_cons_mat.T

    # Point reconciled forecasts
    rhs = jnp.vstack([
        - cons_mat @ base.T,
        jnp.zeros((immutable.shape[0], base.shape[0]))
    ])
    lhs = compl_cons_mat @ cov_cons
    lm = lin_sys(lhs=lhs, rhs=rhs, solver=solver)
    reco = base + (cov_cons @ lm).T
    return reco
