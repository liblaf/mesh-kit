from pathlib import Path

import mkit
import numpy as np
import pyvista as pv
import trimesh as tm
from icecream import ic
from mkit.ext import ICTFaceKit


class Config(mkit.cli.BaseConfig):
    mesh: Path


def main(cfg: Config) -> None:
    target: pv.PolyData = pv.read(cfg.mesh, progress_bar=True)
    source: pv.PolyData = ICTFaceKit().narrow_face
    source.scale(target.length / source.length, inplace=True)
    source.rotate_x(-90, inplace=True)
    source.translate(
        np.asarray(target.center) - np.asarray(source.center), inplace=True
    )
    source.save("source.ply")
    matrix, transformed, cost = tm.registration.icp(
        source.points, target.points, max_iterations=500
    )
    ic(cost)
    source.points = transformed
    source.save("aligned.ply")


if __name__ == "__main__":
    mkit.cli.run(main)
