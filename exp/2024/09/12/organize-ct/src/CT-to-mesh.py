from pathlib import Path

import pyvista as pv

import mkit


class Config(mkit.cli.BaseConfig):
    dpath: Path


def main(cfg: Config) -> None:
    dataset: mkit.io.DICOMDataset = mkit.io.DICOMDataset(cfg.dpath)
    for patient in dataset.values():
        for acq in patient:
            ct: pv.ImageData = acq.ct
            face: pv.PolyData = contour(ct, -200.0)
            skull: pv.PolyData = contour(ct, 200.0)
            dpath: Path = Path("data") / patient.meta.id / acq.meta.date
            dpath.mkdir(parents=True, exist_ok=True)
            face.save(dpath / "00-face.ply")
            skull.save(dpath / "00-skull.ply")


def contour(ct: pv.ImageData, threshold: float) -> pv.PolyData:
    ct = ct.gaussian_smooth(progress_bar=True)
    contour: pv.PolyData = ct.contour([threshold], progress_bar=True)  # pyright: ignore [reportArgumentType]
    contour = contour.connectivity("largest", progress_bar=True)
    return contour


mkit.cli.auto_run()(main)
