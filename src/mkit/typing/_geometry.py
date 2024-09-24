from collections.abc import Mapping
from typing import Any

import numpy as np
import pyvista as pv
from jaxtyping import Shaped

from mkit.typing.array import ArrayLike

# TODO: Better typing
AnyPointCloud = Any
AnyTriMesh = Any
AnyQuadMesh = Any
AnyTetMesh = Any
AnySurfaceMesh = AnyTriMesh | AnyQuadMesh
AnyVolumeMesh = AnyTetMesh
AnyMesh = AnySurfaceMesh | AnyVolumeMesh
AnyPointSet = AnyPointCloud | AnyMesh

AttributesLike = Mapping[str, Shaped[ArrayLike, "N ..."]] | pv.DataSetAttributes
AttributeArray = Shaped[np.ndarray, "N ..."]
