from pathlib import Path

import numpy as np
import pandas as pd
import pydantic
import pyvista as pv
import scipy
import scipy.spatial
import trimesh
from loguru import logger

import mkit


class Config(mkit.cli.BaseConfig):
    result_dir: pydantic.DirectoryPath = Path("data/simulate/stable-neo-hookean/")
    data_dir: pydantic.DirectoryPath = Path("data/patients/")
    output_dir: Path = Path("data/eval/stable-neo-hookean/")
    dataset_dir: pydantic.DirectoryPath = Path("/home/liblaf/Documents/CT/")


def main(cfg: Config) -> None:
    dataset = mkit.io.DICOMDataset(cfg.dataset_dir)
    summary = pd.DataFrame(
        columns=["patient", "dist (mean)", "dist (95)", "dist (99)", "dist (max)"]
    )
    for patient in dataset.values():
        result_fpath: Path = cfg.result_dir / patient.id / "predict.vtu"
        gt_path: Path = (
            cfg.data_dir / patient.id / str(patient[-1].date) / "02-face-non-rigid.vtp"
        )
        if not result_fpath.exists():
            logger.warning("Result not found: {}", result_fpath)
            continue
        result: pv.UnstructuredGrid = pv.read(result_fpath)
        gt: pv.PolyData = mkit.io.pyvista.load_poly_data(gt_path)
        surface = result.extract_surface()
        mkit.io.save(surface, cfg.output_dir / patient.id / "pre-face.vtp")
        result.warp_by_vector("solution", inplace=True)
        surface: pv.PolyData = result.extract_surface()
        surface: pv.PolyData = surface.extract_points(
            surface.point_data["face_mask"]
        ).extract_surface()
        validation_mask = surface.point_data["Weight"] > 0.5
        kdtree = scipy.spatial.KDTree(gt.points)
        _dist, idx = kdtree.query(surface.points[validation_mask])
        surface.point_data["dist"] = np.zeros((surface.n_points,))
        dist = trimesh.util.row_norm(
            (surface.points[validation_mask] - gt.points[idx]) * gt.point_normals[idx]
        )
        summary.loc[len(summary)] = [
            patient.id,
            dist.mean(),
            np.percentile(dist, 95),
            np.percentile(dist, 99),
            dist.max(),
        ]
        surface.point_data["dist"][validation_mask] = dist
        mkit.io.save(surface, cfg.output_dir / patient.id / "post-face-predict.vtp")
        mkit.io.save(gt, cfg.output_dir / patient.id / "post-face-gt.vtp")
        mkit.io.save(
            pv.read(
                cfg.data_dir
                / patient.id
                / str(patient[-1].date)
                / "02-mandible-non-rigid.vtp"
            ),
            cfg.output_dir / patient.id / "post-mandible.vtp",
        )
        mkit.io.save(
            pv.read(
                cfg.data_dir
                / patient.id
                / str(patient[-1].date)
                / "02-maxilla-non-rigid.vtp"
            ),
            cfg.output_dir / patient.id / "post-maxilla.vtp",
        )
        mkit.io.save(
            pv.read(
                cfg.data_dir
                / patient.id
                / str(patient[0].date)
                / "02-mandible-non-rigid.vtp"
            ),
            cfg.output_dir / patient.id / "pre-mandible.vtp",
        )
        mkit.io.save(
            pv.read(
                cfg.data_dir
                / patient.id
                / str(patient[0].date)
                / "02-maxilla-non-rigid.vtp"
            ),
            cfg.output_dir / patient.id / "pre-maxilla.vtp",
        )
    summary.to_csv(cfg.output_dir / "summary.csv")


mkit.cli.auto_run()(main)
