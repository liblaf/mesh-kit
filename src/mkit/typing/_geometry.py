from typing import Any

# TODO: Better typing
AnyPointCloud = Any
AnyTriMesh = Any
AnyQuadMesh = Any
AnyTetMesh = Any
AnySurfaceMesh = AnyTriMesh | AnyQuadMesh
AnyVolumeMesh = AnyTetMesh
AnyMesh = AnySurfaceMesh | AnyVolumeMesh
AnyPointSet = AnyPointCloud | AnyMesh
