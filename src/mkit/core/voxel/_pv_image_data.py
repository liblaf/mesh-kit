from typing import Self

import pyvista as pv

from mkit.core import VoxelGridBase


class PyvistaImageDataMixin(VoxelGridBase):
    @property
    def pyvista_image_data(self) -> pv.ImageData:
        return self._data

    def gaussian_smooth(
        self,
        radius_factor: float = 1.5,
        std_dev: float = 2.0,
        scalars: str | None = None,
    ) -> Self:
        result: pv.ImageData = self._data.gaussian_smooth(
            radius_factor=radius_factor, std_dev=std_dev, scalars=scalars
        )  # pyright: ignore [reportAssignmentType]
        return self.__class__.from_pyvista(result)
