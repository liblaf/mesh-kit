from pathlib import Path
from typing import Self

import mkit.typing as mt
from mkit.core.voxel import VoxelGrid


class DICOM(VoxelGrid):
    @classmethod
    def load(cls, path: mt.StrPath, ext: str | None = ".dcm") -> Self:
        path: Path = Path(path)
        self: Self = super().load(path, ext)
        self._path = path
        return self
