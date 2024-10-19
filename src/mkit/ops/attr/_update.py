import pyvista as pv

import mkit.math as mm
import mkit.typing as mt


def update_attrs(
    attrs: pv.DataSetAttributes, update: mt.AttrsLike | None = None
) -> pv.DataSetAttributes:
    if not update:
        return attrs
    attrs.update({k: mm.as_numpy(v) for k, v in update.items()})
    return attrs
