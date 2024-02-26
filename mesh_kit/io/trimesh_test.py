import pathlib
import tempfile

import numpy as np
import trimesh
from numpy import typing as npt

from mesh_kit.io import trimesh as _io


def test_read() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        mesh_desired: trimesh.Trimesh = trimesh.creation.icosphere()
        mesh_desired.export(tmp / "mesh.ply")
        rng: np.random.Generator = np.random.default_rng()
        mesh_desired.vertex_attributes = {
            "foo": rng.random((mesh_desired.vertices.shape[0],)),
        }
        mesh_desired.face_attributes = {
            "bar": rng.random((mesh_desired.faces.shape[0], 3)),
        }
        data_desired: dict[str, npt.NDArray] = {
            "vert:foo": mesh_desired.vertex_attributes["foo"],
            "face:bar": mesh_desired.face_attributes["bar"],
        }
        np.savez_compressed(tmp / "mesh.npz", **data_desired)
        mesh_actual: trimesh.Trimesh = _io.read(tmp / "mesh.ply")
        np.testing.assert_allclose(mesh_actual.vertices, mesh_desired.vertices)
        np.testing.assert_allclose(mesh_actual.faces, mesh_desired.faces)
        for k, v in mesh_desired.vertex_attributes.items():
            np.testing.assert_allclose(mesh_actual.vertex_attributes[k], v)
        for k, v in mesh_desired.face_attributes.items():
            np.testing.assert_allclose(mesh_actual.face_attributes[k], v)
