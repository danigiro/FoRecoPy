from functools import partial
from typing import Optional

import jax as jax
import jax.numpy as jnp
import scipy.sparse as sps

from forecopy.fun import isDiag, lin_sys
from forecopy.tools import cstools, tetools


class _reconcile:
    def __init__(
        self,
        base: jnp.ndarray,
        cov_mat: jnp.ndarray,
        params: cstools | tetools,
        id_nn: list = [],
    ):
        self.base = base
        self.params = params
        if isDiag(cov_mat) and len(cov_mat.shape) != 1:
            self.cov_mat = cov_mat.diagonal()
        else:
            self.cov_mat = cov_mat
        self.id_nn = id_nn

    def fit(
        self,
        approach="proj",
        solver="default",
        tol=1e-12,
        nn=False,
        immutable: Optional[jnp.ndarray] = None,
    ):
        if immutable is not None:
            if approach not in ["proj", "proj_tol"]:
                raise ValueError(
                    "The 'immutable' option is only available with the 'proj' approach."
                )
        if approach == "proj":
            reco = rproj(
                base=self.base,
                cons_mat=self.params.cons_mat(),
                cov_mat=self.cov_mat,
                immutable=immutable,
                solver=solver,
            )
        elif approach == "proj_tol":
            reco = rproj_tol(
                base=self.base,
                cons_mat=self.params.cons_mat(),
                cov_mat=self.cov_mat,
                immutable=immutable,
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
        strc_cov = (strc_mat.T).multiply(jnp.reciprocal(cov_mat))
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
    out = (strc_mat @ (bts @ jnp.identity(bts.shape[1]))).T
    return out


def rproj_tol(base, cons_mat, cov_mat, immutable: Optional[jnp.ndarray], tol=1e-12):
    if cons_mat is None:
        raise TypeError("Missing required argument: 'cons_mat'")

    if cons_mat.shape[1] != cov_mat.shape[0] or base.shape[1] != cov_mat.shape[0]:
        raise ValueError("The size of the matrices does not match.")

    b_precomputed = jnp.array(cons_mat) @ base.T
    if immutable is not None:
        cons_mat = jnp.vstack([cons_mat, jax.nn.one_hot(immutable, base.shape[1])])
        b_precomputed = jnp.vstack(
            [b_precomputed, jnp.zeros((immutable.size, base.shape[0]))]
        )

    if len(cov_mat.shape) == 1:
        cons_mat = sps.csr_matrix(cons_mat)
        cons_cov = (cons_mat).multiply(cov_mat)
    else:
        cons_cov = cons_mat @ cov_mat

    def matvec_action(y):
        b = b_precomputed @ y
        A = sps.linalg.LinearOperator(
            (b.size, b.size), matvec=lambda v: cons_cov @ (cons_mat.T @ v)
        )
        btilde, _ = sps.linalg.bicgstab(A, b, atol=tol)
        return btilde

    lm = sps.linalg.LinearOperator(
        (cons_mat.shape[0], base.shape[0]), matvec=matvec_action
    )
    lm = lm @ jnp.identity(lm.shape[1])
    out = base - (cons_cov.T @ lm).T
    return out


def sntz(reco, strc_mat, id_nn):
    if strc_mat is None:
        raise TypeError("Missing required argument: 'strc_mat'")
    reco = reco[:, id_nn]
    # reco[reco<0] = 0
    reco = reco.at[jnp.where(reco < 0)].set(0)
    return reco @ strc_mat.T


@partial(jax.jit, static_argnames=["solver"])
def rproj(
    base, cons_mat, cov_mat, immutable: Optional[jnp.ndarray] = None, solver="default"
):
    if cons_mat is None:
        raise TypeError("Missing required argument: 'cons_mat'")

    if cons_mat.shape[1] != cov_mat.shape[0] or base.shape[1] != cov_mat.shape[0]:
        raise ValueError("The size of the matrices does not match.")

    if immutable is not None:
        imm_cons_mat = jax.nn.one_hot(immutable, base.shape[1])
        compl_cons_mat = jnp.vstack([cons_mat, imm_cons_mat])
    else:
        compl_cons_mat = cons_mat

    # check immutable feasibility
    # TODO

    if len(cov_mat.shape) == 1:
        cov_cons = (cov_mat * compl_cons_mat).T
    else:
        cov_cons = cov_mat @ compl_cons_mat.T

    rhs = -cons_mat @ base.T
    if immutable is not None:
        rhs = jnp.vstack([rhs, jnp.zeros((immutable.shape[0], base.shape[0]))])
    # Point reconciled forecasts
    lhs = compl_cons_mat @ cov_cons
    lm = lin_sys(lhs=lhs, rhs=rhs, solver=solver)
    reco = base + (cov_cons @ lm).T
    return reco
