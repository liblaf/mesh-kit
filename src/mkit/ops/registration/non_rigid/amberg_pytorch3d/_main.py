from typing import Any

import pydantic

import mkit


class Params(pydantic.BaseModel):
    distance_threshold: float = 0.1
    eps: float = 1e-4
    gamma: float = 1
    neighbors_count: int = 8
    use_faces: bool = True
    use_vertex_normals: bool = True


def nricp_amberg_pytorch3d(
    source: Any, target: Any, params: Params | None = None
) -> None:
    if params is None:
        params = Params()
    source = mkit.io.trimesh.as_trimesh(source)
    raise NotImplementedError
