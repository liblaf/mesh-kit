from __future__ import annotations

from typing import TYPE_CHECKING

import igl

if TYPE_CHECKING:
    import scipy.sparse
    from numpy.typing import ArrayLike


def grad(
    v: ArrayLike, f: ArrayLike, *, uniform: bool = False
) -> scipy.sparse.csc_matrix:
    """Compute the numerical gradient operator.

    Args:
        v : #v by 3 list of mesh vertex positions
        f : #f by 3 list of mesh face indices [or a #faces by 4 list of tetrahedral indices]
        uniform : boolean (default false). Use a uniform mesh instead of the vertices v

    Returns:
        g : #faces * dim by #v gradient operator
    """
    raise NotImplementedError
    return igl.grad(v, f, uniform)  # pyright: ignore [reportAttributeAccessIssue]
