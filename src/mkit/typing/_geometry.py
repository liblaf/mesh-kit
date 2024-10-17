from collections.abc import Mapping
from typing import Any

import pyvista as pv

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

AttrsLike = Mapping[str, ArrayLike] | pv.DataSetAttributes
