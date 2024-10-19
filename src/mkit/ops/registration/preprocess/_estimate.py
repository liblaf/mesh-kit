from typing import TYPE_CHECKING, Any

import numpy as np
import trimesh.transformations as tf

import mkit
import mkit.typing.numpy as tn

if TYPE_CHECKING:
    import pyvista as pv


def estimate_transform(
    source: Any,
    target: Any,
    init: tn.F44Like | None = None,
) -> tn.F44:
    source: pv.PolyData = mkit.io.pyvista.as_poly_data(source)
    target: pv.PolyData = mkit.io.pyvista.as_poly_data(target)
    init = np.asarray(init) if init is not None else tf.identity_matrix()
    source = source.transform(init, inplace=False)
    trans: tn.F44 = tf.concatenate_matrices(
        mkit.ops.transform.denorm_transformation(target),
        mkit.ops.transform.norm_transformation(source),
        init,
    )
    return trans
