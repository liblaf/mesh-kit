import pooch
import pyvista as pv

REGISTRY: pooch.Pooch = pooch.create(
    pooch.os_cache("mesh-kit"),
    base_url="https://github.com/liblaf/sculptor/releases/download/template/",
    registry={
        "face.ply": "sha256:480a2d7fddcd68361347d9e9d2e199c894015f6092f7d9449afdbba69587870a",
        "skull.ply": "sha256:5efc3f1a70cebdd8e2a6aed4837a833294839239e7363e2b5937ace10423e339",
    },
)


def get_template_face() -> pv.PolyData:
    mesh: pv.PolyData = pv.read(REGISTRY.fetch("face.ply"))
    return mesh


def get_template_skull() -> pv.PolyData:
    mesh: pv.PolyData = pv.read(REGISTRY.fetch("skull.ply"))
    return mesh


def get_template_maxilla() -> pv.PolyData:
    skull: pv.PolyData = get_template_skull()
    bodies: pv.MultiBlock = skull.split_bodies().as_polydata_blocks()
    maxilla: pv.PolyData = bodies[0]
    return maxilla


def get_template_mandible() -> pv.PolyData:
    skull: pv.PolyData = get_template_skull()
    bodies: pv.MultiBlock = skull.split_bodies().as_polydata_blocks()
    mandible: pv.PolyData = bodies[1]
    return mandible
