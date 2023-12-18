import logging
import pathlib
from typing import Annotated

import nrrd
import numpy as np
import trimesh
import typer
from numpy import typing as npt
from scipy import ndimage
from trimesh.voxel import ops

from mesh_kit.common import cli


def find_largest_object(label: npt.NDArray) -> int:
    object_slices: list[tuple[slice]] = ndimage.find_objects(label)
    return (
        np.argmax(
            [
                np.prod([slice_.stop - slice_.start for slice_ in obj_slice])
                for obj_slice in object_slices
            ]
        )
        + 1
    )


def main(
    ct_path: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    output_path: Annotated[pathlib.Path, typer.Argument(dir_okay=False, writable=True)],
    threshold: Annotated[int, typer.Option()] = 0,
) -> None:
    data: npt.NDArray
    header: nrrd.NRRDHeader
    data, header = nrrd.read(str(ct_path))

    def export_voxel(voxel: npt.NDArray, output_path: pathlib.Path) -> None:
        mesh: trimesh.Trimesh = ops.matrix_to_marching_cubes(voxel)
        mesh.apply_scale(header["space directions"])
        mesh.apply_translation(header["space origin"])
        mesh.export(str(output_path))

    data = data > threshold
    label: npt.NDArray[np.int32]
    num_features: int

    # remove background
    label, num_features = ndimage.label(
        ~data,
        structure=ndimage.generate_binary_structure(rank=3, connectivity=3),
    )
    logging.info("Num Features: %d", num_features)
    data = label != find_largest_object(label)
    # data = ndimage.binary_closing(
    #     data,
    #     structure=ndimage.generate_binary_structure(rank=3, connectivity=3),
    #     iterations=2,
    # )

    # # find largest object
    # label, num_features = ndimage.label(
    #     data,
    #     structure=ndimage.generate_binary_structure(rank=3, connectivity=3),
    # )
    # logging.info("Num Features: %d", num_features)
    # data = label == find_largest_object(label)

    # # remove background
    # label, num_features = ndimage.label(
    #     ~data,
    #     structure=ndimage.generate_binary_structure(rank=3, connectivity=1),
    # )
    # logging.info("Num Features: %d", num_features)
    # data = label != find_largest_object(label)

    export_voxel(data, output_path)


if __name__ == "__main__":
    cli.run(main)
