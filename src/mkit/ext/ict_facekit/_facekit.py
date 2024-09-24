from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pooch
import pyvista as pv

import mkit.ext.ict_facekit._topology as t

_REGISTRY: pooch.Pooch = pooch.create(
    pooch.os_cache("mesh-kit"),
    "https://github.com/ICT-VGL/ICT-FaceKit/raw/master/FaceXModel/",
    registry={
        "generic_neutral_mesh.obj": "sha256:eedbc2576d8e5ea57f55255b8f98263213a1efb5431d8bfceed1d7aef10271f9"
    },
)


class ICTFaceKit(Mapping[str, pv.PolyData]):
    mesh: pv.PolyData

    def __init__(self, *, progress_bar: bool = True) -> None:
        filename: Path = Path(
            _REGISTRY.fetch("generic_neutral_mesh.obj", progressbar=progress_bar)
        )
        self.mesh: pv.PolyData = pv.read(filename, progress_bar=progress_bar)  # pyright: ignore [reportAttributeAccessIssue]
        self.mesh.clean(inplace=True, progress_bar=progress_bar)
        self.mesh.point_data["original_point_id"] = np.arange(self.mesh.n_points)
        self.mesh.cell_data["original_cell_id"] = np.arange(self.mesh.n_cells)

    def __getitem__(self, name: str) -> pv.PolyData:
        return self.extract(name)

    def __iter__(self) -> Iterator[str]:
        for g in t.GEOMETRIES:
            yield g.name

    def __len__(self) -> int:
        return len(t.GEOMETRIES)

    def extract(self, *name: str, progress_bar: bool = True) -> pv.PolyData:
        polygons: pv.UnstructuredGrid = self.mesh.extract_cells(
            t.polygon_indices(*name), progress_bar=progress_bar
        )
        surface: pv.PolyData = polygons.extract_surface(progress_bar=progress_bar)
        return surface

    @property
    def face(self) -> pv.PolyData:
        return self.extract("Face")

    @property
    def narrow_face(self) -> pv.PolyData:
        return self.extract("Narrow face area")

    @property
    def head(self) -> pv.PolyData:
        return self.extract("Head")

    @staticmethod
    def vertex_indices(*name: str) -> npt.NDArray[np.integer]:
        return t.vertex_indices(*name)

    @staticmethod
    def polygon_indices(*name: str) -> npt.NDArray[np.integer]:
        return t.polygon_indices(*name)
