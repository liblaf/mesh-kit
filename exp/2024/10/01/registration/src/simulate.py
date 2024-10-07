from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pydantic
import pyvista as pv
from loguru import logger

import mkit
from mkit.physics import Problem
from mkit.physics.energy import elastic

if TYPE_CHECKING:
    import scipy.optimize


class Config(mkit.cli.BaseConfig):
    dataset: pydantic.DirectoryPath = Path("/home/liblaf/Documents/CT/")
    data_dir: Path = Path("data/tetgen/")
    output_dir: Path = Path("data/simulate/")
    material: str = "stable-neo-hookean"


E: float = 1e6  # Young's modulus
nu: float = 0.46  # Poisson's ratio
lambda_: float = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lamé's first parameter
G: float = E / (2 * (1 + nu))  # Shear modulus


def main(cfg: Config) -> None:
    dataset: mkit.io.DICOMDataset = mkit.io.DICOMDataset(cfg.dataset)
    for patient in dataset.values():
        mesh_fpath: Path = cfg.data_dir / patient.id / "pre.vtu"
        output_fpath: Path = cfg.output_dir / cfg.material / patient.id / "predict.vtu"
        if not mesh_fpath.exists():
            logger.warning("Mesh not found: {}", mesh_fpath)
            continue
        if output_fpath.exists():
            logger.info("Output exists: {}", output_fpath)
            continue
        mesh: pv.UnstructuredGrid = pv.read(mesh_fpath)
        material: elastic.Material = elastic.get_preset(cfg.material)
        material.cell_data.update({"mu": G, "lambda": lambda_})
        mesh.cell_data.update(material.cell_data)  # pyright: ignore [reportArgumentType]
        problem: Problem = Problem(mesh, material.energy_fn)
        res: scipy.optimize.OptimizeResult = problem.solve()
        logger.info("{}", res)
        disp: npt.NDArray[np.floating] = problem.make_disp(res.x)
        mesh.point_data["solution"] = disp
        mesh.cell_data["energy_density"] = np.asarray(
            problem.model.energy_density(disp)
        )
        mesh.field_data["execution_time"] = res["execution_time"]
        mesh.field_data["success"] = res["success"]
        mkit.io.save(mesh, output_fpath)


if __name__ == "__main__":
    mkit.cli.run(main)
