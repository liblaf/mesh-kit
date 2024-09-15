from pathlib import Path
from typing import Any

import mkit
import numpy as np
import pyvista as pv
import trimesh as tm
from icecream import ic


class CLIConfig(mkit.cli.CLIBaseConfig):
    predict: Path
    ground_truth: Path
    output: Path


def main(cfg: CLIConfig) -> None:
    predict: pv.UnstructuredGrid = pv.read(cfg.predict)
    gt: pv.PolyData = pv.read(cfg.ground_truth)
    surface: pv.PolyData = predict.extract_surface(progress_bar=True)
    bodies: pv.MultiBlock = surface.split_bodies()
    predict_face: pv.PolyData = bodies[1].extract_surface()
    predict_face.warp_by_vector("solution", inplace=True)
    predict_face = mkit.ops.transfer.surface_to_surface(
        gt, predict_face, point_data_names=["validation"]
    )
    eval_face = evaluate(predict_face, gt)
    distance = eval_face.point_data["distance"]
    ic(distance.max())
    ic(distance.mean())
    ic(np.percentile(distance, 95))
    eval_face.save(cfg.output)


def evaluate(predict: Any, gt: Any) -> pv.PolyData:
    predict: pv.PolyData = mkit.io.as_polydata(predict)
    gt: tm.Trimesh = mkit.io.as_trimesh(gt)
    validation = np.asarray(predict.point_data["validation"], bool)
    sdist = gt.nearest.signed_distance(predict.points)
    predict.point_data["signed_distance"] = sdist
    predict.point_data["signed_distance"][~validation] = 0
    _closest, dist, _triangle_id = gt.nearest.on_surface(predict.points)
    predict["distance"] = dist
    predict.point_data["distance"][~validation] = 0
    return predict


if __name__ == "__main__":
    mkit.cli.run(main)
