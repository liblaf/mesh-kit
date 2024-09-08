from typing import Any

import numpy as np
import numpy.typing as npt
import pyvista as pv
from icecream import ic

mesh = pv.read("~/.cache/ubelt/generic_neutral_mesh.obj")
ic(mesh)
