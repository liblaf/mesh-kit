from typing import NamedTuple

import numpy as np
from loguru import logger

import mkit.typing.numpy as nt

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
    Geometry("Mouth socket", np.s_[11248:13294], np.s_[11144:13226]),
    Geometry("Eye socket left", np.s_[13294:13678], np.s_[13226:13630]),
    Geometry("Eye socket right", np.s_[13678:14062], np.s_[13630:14034]),
    Geometry("Gums and tongue", np.s_[14062:17039], np.s_[14034:17006]),
    Geometry("Teeth", np.s_[17039:21451], np.s_[17006:21496]),
    Geometry("Eyeball left", np.s_[21451:23021], np.s_[21496:23094]),
    Geometry("Eyeball right", np.s_[23021:24591], np.s_[23094:24692]),
    Geometry("Lacrimal fluid left", np.s_[24591:24795], np.s_[24692:24855]),
    Geometry("Lacrimal fluid right", np.s_[24795:24999], np.s_[24855:25018]),
    Geometry("Eye blend left", np.s_[24999:25023], np.s_[25018:25033]),
    Geometry("Eye blend right", np.s_[25023:25047], np.s_[25033:25048]),
    Geometry("Eye occlusion left", np.s_[25047:25200], np.s_[25048:25176]),
    Geometry("Eye occlusion right", np.s_[25200:25351], np.s_[25176:25304]),
    Geometry("Eyelashes left", np.s_[25351:26035], np.s_[25304:25844]),
    Geometry("Eyelashes right", np.s_[26035:26719], np.s_[25844:26384]),
    # Face Area Details
    Geometry("Full face area", np.s_[0:9409], np.s_[0:9230]),
    Geometry("Narrow face area", np.s_[0:6706], np.s_[0:6560]),
    # Custom
    Geometry("Head", np.s_[0:11248], np.s_[0:11144]),
]


def vertex_indices(*name: str) -> nt.IN:
    logger.warning("Vertex indices is not reliable!")
    indices: nt.IN = np.concatenate([
        np.r_[g.vertex_indices] for g in GEOMETRIES if g.name in name
    ])
    return indices


def polygon_indices(*name: str) -> nt.IN:
    indices: nt.IN = np.concatenate([
        np.r_[g.polygon_indices] for g in GEOMETRIES if g.name in name
    ])
    return indices
