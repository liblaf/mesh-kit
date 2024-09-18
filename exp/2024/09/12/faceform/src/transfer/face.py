import pyvista as pv
import scipy.spatial

import mkit
import mkit.typing.numpy as nt


class Config(mkit.cli.BaseConfig):
    pass


def main(cfg: Config) -> None:
    source: pv.PolyData = mkit.io.pyvista.read_poly_data("data/registration/face.obj")
    target: pv.PolyData = mkit.ext.sculptor.get_template_face()
    target = transfer_cell_data(source, target)
    target.field_data["GroupNames"] = source.field_data["GroupNames"]
    target.save("data/transfer/face.vtp")


def transfer_cell_data(source: pv.PolyData, target: pv.PolyData) -> pv.PolyData:
    source_cell_centers: pv.PolyData = source.cell_centers()
    target_cell_centers: pv.PolyData = target.cell_centers()
    kdtree: scipy.spatial.KDTree = scipy.spatial.KDTree(source_cell_centers.points)
    idx: nt.IN
    _dist, idx = kdtree.query(target_cell_centers.points)
    target.cell_data["GroupIds"] = source_cell_centers.point_data["GroupIds"][idx]
    return target


mkit.cli.auto_run()(main)
