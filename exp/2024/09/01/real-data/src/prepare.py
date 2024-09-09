from pathlib import Path

import numpy as np
import pyvista as pv
from icecream import ic
from sympy import inverse_laplace_transform

import mkit
from mkit.typing import StrPath


class CLIConfig(mkit.cli.CLIBaseConfig): ...


def main(cfg: CLIConfig) -> None:
    data_dir: Path = Path("~/Documents/data/targets/120056").expanduser()
    face_pre: pv.PolyData = read_vtu_as_polydata(data_dir / "pre/99-face.vtu")
    mandible_pre: pv.PolyData = read_vtu_as_polydata(
        data_dir / "pre/99-mandible.vtu", flip=True
    )
    maxilla_pre: pv.PolyData = read_vtu_as_polydata(
        data_dir / "pre/99-maxilla.vtu", flip=True
    )
    face_post: pv.PolyData = read_vtu_as_polydata(data_dir / "post/99-face.vtu")
    mandible_post: pv.PolyData = read_vtu_as_polydata(
        data_dir / "post/99-mandible.vtu", flip=True
    )
    maxilla_post: pv.PolyData = read_vtu_as_polydata(
        data_dir / "post/99-maxilla.vtu", flip=True
    )
    mandible_pre.point_data["pin_disp"] = mandible_post.points - mandible_pre.points
    mandible_pre.point_data["pin_mask"] = True  # pyright: ignore [reportArgumentType]
    maxilla_pre.point_data["pin_disp"] = maxilla_post.points - maxilla_pre.points
    maxilla_pre.point_data["pin_mask"] = True  # pyright: ignore [reportArgumentType]
    face_pre.point_data["pin_disp"] = np.zeros((face_pre.n_points, 3))
    face_pre.point_data["pin_mask"] = False  # pyright: ignore [reportArgumentType]
    pre_tri: pv.PolyData = pv.merge(
        [face_pre, mandible_pre, maxilla_pre], progress_bar=True
    )
    ic(pre_tri.point_data)
    pre_tet: pv.UnstructuredGrid = mkit.ext.tetwild(pre_tri)
    pre_tet = mkit.ops.transfer.surface_to_tetmesh(pre_tri, pre_tet)
    pre_surface: pv.PolyData = pre_tet.extract_surface(progress_bar=True)
    bodies: pv.MultiBlock = pre_surface.split_bodies(progress_bar=True)
    pre_skull = ug_to_polydata(bodies[0])
    pre_face = ug_to_polydata(bodies[1])
    pre_skull.save("data/pre-skull.vtp")
    pre_face.save("data/pre-face.vtp")
    face_post.save("data/post-face-gt.vtp")
    # post_skull = pre_skull.warp_by_vector("pin_disp")
    # post_skull.save(("data/post-skull.vtp"))
    pv.merge([mandible_post, maxilla_post]).save("data/post-skull-gt.vtp")
    pre_tet.save("data/pre.vtu")


def read_vtu_as_polydata(path: StrPath, *, flip: bool = False) -> pv.PolyData:
    ug: pv.UnstructuredGrid = pv.read(path)
    surface: pv.PolyData = ug_to_polydata(ug)
    if flip:
        surface.flip_normals()
    return surface


def ug_to_polydata(ug: pv.UnstructuredGrid) -> pv.PolyData:
    surface: pv.PolyData = ug.extract_surface(progress_bar=True)
    del surface.point_data["vtkOriginalPointIds"]
    del surface.cell_data["vtkOriginalCellIds"]
    return surface


if __name__ == "__main__":
    mkit.cli.run(main)
