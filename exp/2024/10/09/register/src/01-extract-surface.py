import concurrent.futures as cf
from pathlib import Path
from typing import TYPE_CHECKING

import pydantic

import mkit.io as mi
import mkit.struct as ms
import mkit.utils as mu

if TYPE_CHECKING:
    import pyvista as pv


class Config(mu.cli.BaseConfig):
    dataset: pydantic.DirectoryPath = Path("~/data/01-raw/01-CT").expanduser()
    output: Path = Path("~/data/01-raw/02-surface").expanduser()


def process(acquisition: ms.DICOM, output: Path) -> None:
    with mu.timer(f"{acquisition.patient_id} {acquisition.acquisition_date}"):
        face: pv.PolyData = acquisition.extract_face()
        skull: pv.PolyData = acquisition.extract_skull()
        mi.save(output / "face.ply", face)
        mi.save(output / "skull.ply", skull)
        mu.save_pydantic(output / "acquisition.json", acquisition.meta)


def main(cfg: Config) -> None:
    dataset: ms.DICOMDataset = ms.DICOMDataset(cfg.dataset)
    with cf.ProcessPoolExecutor(max_workers=8) as executor:
        futures: list[cf.Future[None]] = []
        for patient in dataset.values():
            patient_dpath: Path = cfg.output / patient.patient_id
            for acquisition in patient:
                out_dpath: Path = patient_dpath / ms.dicom_dataset.format_date(
                    acquisition.acquisition_date
                )
                futures.append(executor.submit(process, acquisition, out_dpath))
                mu.save_pydantic(out_dpath / "acquisition.json", acquisition.meta)
            mu.save_pydantic(patient_dpath / "patient.json", patient.meta)
        cf.wait(futures)


mu.cli.auto_run()(main)
