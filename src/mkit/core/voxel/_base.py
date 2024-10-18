import pyvista as pv

import mkit


class VoxelGridBase(mkit.DataObject[pv.ImageData]):
    _data: pv.ImageData
