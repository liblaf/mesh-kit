import pyvista as pv
import scipy.spatial

import mkit
import mkit.typing.numpy as nt


class Config(mkit.cli.BaseConfig):
    pass


def main(cfg: Config) -> None:
    source: pv.PolyData = mkit.io.pyvista.read_poly_data(
        "data/registration/mandible.obj"
    )
    target: pv.PolyData = mkit.ext.sculptor.get_template_mandible()
    target = transfer_cell_data(source, target)
    target.save("data/transfer/mandible.vtp")


def transfer_cell_data(source: pv.PolyData, target: pv.PolyData) -> pv.PolyData:
    source_cell_centers: pv.PolyData = source.cell_centers()
    target_cell_centers: pv.PolyData = target.cell_centers()
    kdtree: scipy.spatial.KDTree = scipy.spatial.KDTree(source_cell_centers.points)
    idx: nt.IN
    _dist, idx = kdtree.query(target_cell_centers.points)
    target.cell_data["GroupIds"] = source_cell_centers.point_data["GroupIds"][idx]
    return target


mkit.cli.auto_run()(main)
