from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from loguru import logger

"""
### Face Model Topology

| Ordinal#| Geometry name        | Vertex indices | Polygon indices | #Vertices | #Faces |
|---------|----------------------|----------------|-----------------|-----------|--------|
| n/a     | All                  | [0:26718]      | [0:26383]       | 26719     | 26384  |
| #0      | Face                 | [0:9408]       | [0:9229]        | 9409      | 9230   |
| #1      | Head and Neck        | [9409:11247]   | [9230:11143]    | 1839      | 1914   |
| #2      | Mouth socket         | [11248:13293]  | [11144:13225]   | 2046      | 2082   |
| #3      | Eye socket left      | [13294:13677]  | [13226:13629]   | 384       | 404    |
| #4      | Eye socket right     | [13678:14061]  | [13630:14033]   | 384       | 404    |
| #5      | Gums and tongue      | [14062:17038]  | [14034:17005]   | 2977      | 2972   |
| #6      | Teeth                | [17039:21450]  | [17006:21495]   | 4412      | 4490   |
| #7      | Eyeball left         | [21451:23020]  | [21496:23093]   | 1570      | 1598   |
| #8      | Eyeball right        | [23021:24590]  | [23094:24691]   | 1570      | 1598   |
| #9      | Lacrimal fluid left  | [24591:24794]  | [24692:24854]   | 204       | 163    |
| #10     | Lacrimal fluid right | [24795:24998]  | [24855:25017]   | 204       | 163    |
| #11     | Eye blend left       | [24999:25022]  | [25018:25032]   | 24        | 15     |
| #12     | Eye blend right      | [25023:25046]  | [25033:25047]   | 24        | 15     |
| #13     | Eye occlusion left   | [25047:25198]  | [25048:25175]   | 152       | 128    |
| #14     | Eye occlusion right  | [25199:25350]  | [25176:25303]   | 152       | 128    |
| #15     | Eyelashes left       | [25351:26034]  | [25304:25843]   | 684       | 540    |
| #16     | Eyelashes right      | [26035:26718]  | [25844:26383]   | 684       | 540    |

### Face Area Details

| Ordinal#| Geometry name    | Vertex indices | Polygon indices | #Vertices | #Faces |
|---------|------------------|----------------|-----------------|-----------|--------|
| #0      | Full face area   | [0:9408]       | [0:9229]        | 9409      | 9230   |
| #1      | Narrow face area | [0:6705]       | [0:6559]        | 6706      | 6560   |
"""


class Geometry(NamedTuple):
    name: str
    vertex_indices: slice
    polygon_indices: slice


GEOMETRIES: list[Geometry] = [
    # Face Model Topology
    Geometry("Face", np.s_[0:9409], np.s_[0:9230]),
    Geometry("Head and Neck", np.s_[9409:11248], np.s_[9230:11144]),
    # Face Area Details
    Geometry("Full face area", np.s_[0:9409], np.s_[0:9230]),
    Geometry("Narrow face area", np.s_[0:6706], np.s_[0:6560]),
    # Custom
    Geometry("Head", np.s_[0:11248], np.s_[0:11144]),
]


def vertex_indices(name: str) -> npt.NDArray[np.integer]:
    logger.warning("Vertex indices may be unreliable!", name)
    for g in GEOMETRIES:
        if g.name == name:
            return np.r_[g.vertex_indices]
    raise KeyError(name)


def polygon_indices(name: str) -> npt.NDArray[np.integer]:
    for g in GEOMETRIES:
        if g.name == name:
            return np.r_[g.polygon_indices]
    raise KeyError(name)
