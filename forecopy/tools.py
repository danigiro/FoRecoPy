"""
This module provides **utility classes for forecast reconciliation**, with
support for both cross-sectional and temporal frameworks. The tools in
**FoRecoPy** are built around structured matrices that represent aggregation
relationships and linear constraints, which serve as the foundation for
different reconciliation methods. Two complementary toolsets are available:

- :func:`cstools() <forecopy.tools.cstools>`:
  For hierarchically, grouped, or otherwise linearly constrained series
  observed at the same frequency.

- :func:`tetools() <forecopy.tools.tetools>`:
  For temporal hierarchies, where the same time series can be aggregated
  or linearly combined at multiple frequencies (e.g., monthly, quarterly,
  yearly). This allows one to reconcile forecasts across frequencies so that,
  for example, the sum of 12 monthly forecasts matches the corresponding
  annual forecast.

"""

from typing import Optional

import jax.numpy as jnp

from forecopy.fun import factors


class cstools:
    """
    Cross-sectional reconciliation tools
    ------------------------------------
    Utilities for working with linearly constrained (hierarchical/grouped)
    multivariate time series in a cross-sectional framework.

    This class encapsulates standard matrices used in forecast reconciliation:
    the aggregation matrix :math:`\\mathbf{A}`, the structural matrix
    :math:`\\mathbf{S}`, and the zero-constraints matrix :math:`\\mathbf{C}`.
    Given either :math:`\\mathbf{A}` or :math:`\\mathbf{C}` at construction,
    the remaining matrices and dimension metadata are derived.

    Parameters
    ----------

    ``agg_mat``: ndarray
        A (:math:`n_a \\times n_b`) numeric matrix representing the
        cross-sectional aggregation matrix. It maps the :math:`n_b`
        bottom-level (free) variables into the :math:`n_a` upper
        (constrained) variables.

    ``cons_mat``: ndarray
        A (:math:`n_a \\times n`) numeric matrix representing the
        cross-sectional zero constraints. It spans the null space
        for the reconciled forecasts.

    Attributes
    ----------
    ``agg_mat``: ndarray
        The aggregation matrix :math:`\\mathbf{A}`
        (may be inferred from :math:`\\mathbf{C}`).
    ``dim``: tuple
        Dimension summary:

        * if :math:`\\mathbf{A}` given: :math:`(n, n_a, n_b)` with
          :math:`n = n_a + n_b`.
        * if :math:`\\mathbf{C}` given: :math:`(n, r, n - r)` with
          :math:`r = \\text{ rank/rows of } \\mathbf{C}`.

    ``_cons_mat``: ndarray
        The constraints matrix :math:`\\mathbf{C}`, built on demand
        if :math:`\\mathbf{A}` is provided.

    ``_strc_mat``: ndarray
        The structural matrix
        :math:`\\mathbf{S} = [\\mathbf{A}'\\quad\\mathbf{I}_{n_b}]'`.

    Methods
    -------

    ``strc_mat()`` -> jnp.ndarray
        Returns the temporal structural matrix :math:`\\mathbf{S}`.
        Requires `agg_mat` to be available.

    ``cons_mat()`` -> jnp.ndarray
        Returns the temporal constraints matrix :math:`\\mathbf{C}`.

    Notes
    -----

    - Shapes:

      * :math:`\\mathbf{A}` has shape :math:`(n_a, n_b)`.
      * :math:`\\mathbf{S}` has shape :math:`(n_a + n_b, n_b) = (n, n_b)`.
      * :math:`\\mathbf{C}` has shape :math:`(n_a, n) = (n_a, n_a + n_b)`
        when built from :math:`\\mathbf{A}`.

    - When initialized with `cons_mat`, :math:`\\mathbf{A}` is only inferred
      if the leading :math:`(r\\times r)` block equals the identity;
      specifically, if :math:`\\mathbf{C} = [\\mathbf{I}_{r}\\quad-\\mathbf{A}]`.

    See Also
    --------
    :func:`cscov <forecopy.cov.cscov>`
    :func:`csrec <forecopy.lsrec.csrec>`

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import forecopy as rpy
    >>> # One-level hierarchy: Z = X + Y
    >>> A = jnp.array([[1., 1.]])            # shape (n_a=1, n_b=2)
    >>> tools = rpy.cstools(agg_mat=A)
    >>> tools.dim
    (3, 1, 2)
    >>> S = tools.strc_mat()
    >>> S.shape                              # [A' ; I_2]'
    (3, 2)
    >>> C = tools.cons_mat()
    >>> C                                    # [I_1  -A]
    Array([[ 1., -1., -1.]], dtype=float32)
    >>>
    >>> # Start from constraints: C = [1, -1, -1]
    >>> C = jnp.array([[1., -1., -1.]])      # r=1, n=3
    >>> tools2 = rpy.cstools(cons_mat=C)
    >>> tools2.dim
    (3, 1, 2)
    >>> tools2.agg_mat                       # inferred because C = [I | -A]
    Array([[1., 1.]], dtype=float32)
    >>> tools2.strc_mat().shape
    (3, 2)
    """

    def __init__(
        self,
        agg_mat: Optional[jnp.ndarray] = None,
        cons_mat: Optional[jnp.ndarray] = None,
    ):
        self.agg_mat = agg_mat
        self._cons_mat = cons_mat
        self._strc_mat = None
        self.dim = ()

        if agg_mat is not None:
            n = sum(agg_mat.shape)
            self.dim = (n,) + agg_mat.shape
        elif cons_mat is not None:
            r = cons_mat.shape[0]
            self.dim = (cons_mat.shape[1], r, cons_mat.shape[1] - r)
            if jnp.all(cons_mat[0:r, 0:r] == jnp.eye(r)):
                self.agg_mat = -cons_mat[0:r, r:] + 0
        else:
            raise TypeError("Missing required argument: 'agg_mat' or 'cons_mat'")

    def strc_mat(self):
        if self.agg_mat is not None:
            if self._strc_mat is None:
                self._strc_mat = jnp.vstack(
                    (self.agg_mat, jnp.eye(self.agg_mat.shape[1]))
                )

            return self._strc_mat
        else:
            raise TypeError(
                "A structural representation is not available for this constraints matrix"
            )

    def cons_mat(self):
        if self._cons_mat is None:
            self._cons_mat = jnp.hstack((jnp.eye(self.agg_mat.shape[0]), -self.agg_mat))

        return self._cons_mat


class tetools:
    """
    Temporal reconciliation tools
    -----------------------------
    Utilities for forecast reconciliation through temporal hierarchies.

    Given a set of temporal aggregation orders, this class constructs the
    temporal aggregation matrix (linear-combination matrix) and provides the
    corresponding structural and zero-constraint matrices used in temporal
    reconciliation.

    Parameters
    ----------

    ``agg_order``: list | int
        Highest available sampling frequency per seasonal cycle (max. order
        of temporal aggregation, :math:`m`), or a list representing a
        subset of :math:`p` factors of :math:`m`.

    ``tew``: str, default `sum`
        Temporal aggregation weighting scheme applied within each high-to-low
        block when building the aggregation matrix:

        * `sum`: sum over the block.
        * `avg`: arithmetic average over the block.
        * `last`: take the last element in the block.
        * `first`: take the first element in the block.

    ``fh``: int, default 1
        Forecast horizon for the lowest frequency (most temporally aggregated)
        series.

    Attributes
    ----------

    ``m``: int
        Maximum aggregation order, `m = max(agg_order)`.

    ``kset``: list[int]
        Decreasing set of temporal aggregation orders that divide `m`. If a
        single integer was supplied to `agg_order`, this equals the full list
        of positive factors of `m`.

    ``p``: int
        Number of elements in `kset`.

    ``ks``: int
        Partial sum of factors :math:`k^\\ast`, defined as `sum(m / kset[:-1])`.

    ``kt``: int
        Total sum of factors :math:`k_t`, defined as `sum(m / kset)`.

    ``tew``: str
        The requested temporal weighting flag (see parameter description).

    ``_agg_mat``: jnp.ndarray
        Temporal aggregation matrix :math:`\\mathbf{A}`.

    ``_strc_mat``: jnp.ndarray or None
        Temporal structural matrix,
        :math:`\\mathbf{S} = [\\mathbf{A}'\\quad\\mathbf{I}_{m}]'`.

    ``_cons_mat``: jnp.ndarray or None
        Temporal zero-constraints matrix,
        :math:`\\mathbf{C} = [\\mathbf{I}_{k^\\ast}\\quad-\\mathbf{A}]`.

    Methods
    -------

    ``strc_mat()`` -> jnp.ndarray
        Return the temporal structural matrix :math:`\\mathbf{S}`.
    ``cons_mat()`` -> jnp.ndarray
        Return the temporal constraints matrix :math:`\\mathbf{C}``.

    Notes
    -----
    - Shapes:
        * :math:`\\mathbf{A}` has shape :math:`(k^\\ast, m)`.
        * :math:`\\mathbf{S}` has shape :math:`(k^\\ast + m, m) = (k_t, m)`.
        * :math:`\\mathbf{C}` has shape :math:`(k^\\ast, k_t)`.

    See Also
    --------
    :func:`tecov <forecopy.cov.tecov>`
    :func:`terec <forecopy.lsrec.terec>`

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import forecopy as rpy
    >>>
    >>> # Quarterly over monthly (m = 4), full factor set {4,2,1}
    >>> obj = rpy.tetools(agg_order=4, tew='sum', fh=1)
    >>> obj.kset
    [4, 2, 1]
    >>> A = obj._agg_mat
    >>> S = obj.strc_mat()
    >>> C = obj.cons_mat()
    >>> A.shape, S.shape, C.shape
    ((3, 4), (7, 4), (3, 7))
    >>>
    >>> # Custom set (divisors of m=12 that you care about): {12, 6, 3, 1}
    >>> obj2 = rpy.tetools(agg_order=[12, 6, 3, 1], fh=2)
    >>> obj2.kset
    [12, 6, 3, 1]
    >>> obj2._agg_mat.shape      # i.e., (2+4+24, 12) = (30, 12)
    ( (12/6)*2 + (12/3)*2 + (12/1)*2, 12 )
    """

    def __init__(self, agg_order: list[int] | int, tew: str = "sum", fh: int = 1):
        self.m = int(max(agg_order)) if isinstance(agg_order, list) else agg_order
        kset_full = factors(self.m)
        if isinstance(agg_order, int):
            kset = kset_full
        else:
            kset = sorted([int(i) for i in agg_order if i in kset_full], reverse=True)
            if min(kset) != 1:
                kset = kset + [1]
        self.kset = kset
        self.p = len(self.kset)
        self.ks = int(sum(self.m / jnp.array(self.kset[0:-1])))
        self.kt = int(sum(self.m / jnp.array(self.kset)))
        self.tew = tew

        if self.tew == "sum":
            weights = [jnp.repeat(1, i) for i in self.kset[0:-1]]
        elif self.tew == "avg":
            weights = [jnp.repeat(1 / i, i) for i in self.kset[0:-1]]
        elif self.tew == "last":
            weights = [jnp.hstack([jnp.repeat(0, i - 1), i]) for i in self.kset[0:-1]]
        elif self.tew == "first":
            weights = [jnp.hstack([i, jnp.repeat(0, i - 1)]) for i in self.kset[0:-1]]
        else:
            raise ValueError("tew")

        freq = self.m / jnp.array(self.kset)
        agg_mat = [
            jnp.kron(jnp.eye(int(freq[i]) * fh), weights[i])
            for i in range((len(freq) - 1))
        ]
        self._agg_mat = jnp.vstack(agg_mat)
        self._cons_mat = jnp.hstack((jnp.eye(self._agg_mat.shape[0]), -self._agg_mat))
        self._strc_mat = jnp.vstack((self._agg_mat, jnp.eye(self._agg_mat.shape[1])))

    def strc_mat(self):
        return self._strc_mat

    def cons_mat(self):
        return self._cons_mat
