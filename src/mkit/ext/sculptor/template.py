import pyvista as pv
import ubelt as ub


def get_template_face() -> pv.PolyData:
    fpath: str = ub.grabdata(
        "https://github.com/liblaf/sculptor/releases/download/template/face.ply",
        hash_prefix="bdb77d8f205c36d5",
        hasher="sha512",
    )
    mesh: pv.PolyData = pv.read(fpath)
    return mesh


def get_template_skull() -> pv.PolyData:
    fpath: str = ub.grabdata(
        "https://github.com/liblaf/sculptor/releases/download/template/skull.ply",
        hash_prefix="ef825bc9a35062a0",
        hasher="sha512",
    )
    mesh: pv.PolyData = pv.read(fpath)
    return mesh


def get_template_maxilla() -> pv.PolyData:
    skull: pv.PolyData = get_template_skull()
    bodies: pv.MultiBlock = skull.split_bodies(progress_bar=True).as_polydata_blocks()
    maxilla: pv.PolyData = bodies[0]
    return maxilla


def get_template_mandible() -> pv.PolyData:
    skull: pv.PolyData = get_template_skull()
    bodies: pv.MultiBlock = skull.split_bodies(progress_bar=True).as_polydata_blocks()
    mandible: pv.PolyData = bodies[1]
    return mandible
