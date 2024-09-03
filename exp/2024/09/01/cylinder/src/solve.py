from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pyvista as pv
from loguru import logger

import mkit
from mkit.physics import Problem
from mkit.physics.energy import elastic

if TYPE_CHECKING:
    import scipy.optimize


class Config(mkit.cli.BaseConfig):
    input: Path
    output: Path
    material: str


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = pv.read(cfg.input)
    material: elastic.Material = elastic.get_preset(cfg.material)
    mesh.cell_data.update(material.cell_data)  # pyright: ignore [reportArgumentType]
    problem: Problem = Problem(mesh, material.energy_fn)
    res: scipy.optimize.OptimizeResult = problem.solve()
    logger.info("{}", res)
    disp: npt.NDArray[np.floating] = problem.make_disp(res.x)
    mesh.point_data["solution"] = disp
    mesh.cell_data["energy_density"] = np.asarray(problem.model.energy_density(disp))
    mesh.field_data["execution_time"] = res["execution_time"]
    mesh.field_data["success"] = res["success"]
    mesh.save(cfg.output)


if __name__ == "__main__":
    mkit.cli.run(main)
