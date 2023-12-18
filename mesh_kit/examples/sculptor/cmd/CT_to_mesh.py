import enum
import logging
import pathlib
from typing import Annotated, Optional

import nrrd
import numpy as np
import trimesh
import typer
from matplotlib import pyplot as plt
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


class Component(enum.Enum):
    FACE: str = "face"
    SKULL: str = "skull"


THRESHOLDS: dict[Component, int] = {
    Component.FACE: 0,
    Component.SKULL: 250,
}


def main(
    ct_path: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    output_path: Annotated[pathlib.Path, typer.Argument(dir_okay=False, writable=True)],
    component: Annotated[Component, typer.Option()],
    record_dir: Annotated[
        Optional[pathlib.Path], typer.Option("--record", exists=True, file_okay=False)
    ] = None,
) -> None:
    counter: int = 0

    def save_img(data: npt.NDArray) -> None:
        nonlocal counter
        if record_dir is None:
            return
        plt.imsave(
            fname=(record_dir / f"{counter:02d}").with_suffix(".png"),
            arr=np.interp(
                data[:, :, data.shape[2] // 2], [data.min(), data.max()], [0, 255]
            ),
        )
        counter += 1

    data: npt.NDArray
    header: nrrd.NRRDHeader
    data, header = nrrd.read(str(ct_path))

    def export_voxel(voxel: npt.NDArray, output_path: pathlib.Path) -> None:
        mesh: trimesh.Trimesh = ops.matrix_to_marching_cubes(voxel)
        mesh.apply_scale(header["space directions"])
        mesh.apply_translation(header["space origin"])
        mesh.export(str(output_path))

    data = data > THRESHOLDS[component]
    label: npt.NDArray[np.int32]
    num_features: int

    match component:
        case Component.FACE:
            # remove background
            label, num_features = ndimage.label(
                ~data,
                structure=ndimage.generate_binary_structure(rank=3, connectivity=3),
            )
            logging.info("Num Features: %d", num_features)
            data = label != find_largest_object(label)
            save_img(data)
            data = ndimage.binary_closing(
                data,
                structure=ndimage.generate_binary_structure(rank=3, connectivity=3),
                iterations=2,
            )
            save_img(data)

            # find largest object
            label, num_features = ndimage.label(
                data,
                structure=ndimage.generate_binary_structure(rank=3, connectivity=1),
            )
            logging.info("Num Features: %d", num_features)
            data = label == find_largest_object(label)
            save_img(data)

            # remove background
            label, num_features = ndimage.label(
                ~data,
                structure=ndimage.generate_binary_structure(rank=3, connectivity=3),
            )
            logging.info("Num Features: %d", num_features)
            data = label != find_largest_object(label)
            save_img(data)

        case Component.SKULL:
            # remove background
            label, num_features = ndimage.label(
                ~data,
                structure=ndimage.generate_binary_structure(rank=3, connectivity=3),
            )
            logging.info("Num Features: %d", num_features)
            data = label != find_largest_object(label)
            save_img(data)

            # find largest object
            label, num_features = ndimage.label(
                data,
                structure=ndimage.generate_binary_structure(rank=3, connectivity=1),
            )
            logging.info("Num Features: %d", num_features)
            data = label == find_largest_object(label)
            save_img(data)

    export_voxel(data, output_path)


if __name__ == "__main__":
    cli.run(main)
