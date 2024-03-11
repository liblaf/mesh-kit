import pathlib
import subprocess
from typing import no_type_check

import meshtaichi_patcher
import numpy as np
import taichi as ti
import trimesh
from trimesh import creation

from mesh_kit.physics import elastic, mtm

E: float = 3000
nu: float = 0.47
ti.init()


def test_force(tmp_path: pathlib.Path) -> None:
    surface: trimesh.Trimesh = creation.box()
    surface.export(tmp_path / "surface.ply", encoding="ascii")
    subprocess.run(["tetgen", "-z", tmp_path / "surface.ply"])
    mesh: ti.MeshInstance = meshtaichi_patcher.load_mesh(
        str(tmp_path / "surface.1.node"), relations=["CV", "CE", "EV"]
    )
    mesh.cells.place({"lambda_": float, "mu": float})
    lambda_: ti.ScalarField = mesh.cells.get_member_field("lambda_")
    lambda_.fill(elastic.E_nu2lambda(E, nu))
    mu: ti.ScalarField = mesh.cells.get_member_field("mu")
    mu.fill(elastic.E_nu2G(E, nu))
    mesh.edges.place({"K": ti.math.mat3})
    mesh.verts.place(
        {
            "pos": ti.math.vec3,
            "K": ti.math.mat3,
            "x": ti.math.vec3,
            "f": ti.math.vec3,
        }
    )
    pos: ti.MatrixField = mesh.verts.get_member_field("pos")
    pos.from_numpy(mesh.get_position_as_numpy())
    mtm.calc_stiffness(mesh)

    @no_type_check
    @ti.kernel
    def _x(mesh: ti.template(), x: ti.template()):
        for v in mesh.verts:
            for i in ti.static(range(3)):
                v.x[i] = x[v.id * 3 + i]

    @no_type_check
    @ti.kernel
    def _Ax(mesh: ti.template(), Ax: ti.template()):
        for v in mesh.verts:
            for i in ti.static(range(3)):
                Ax[v.id * 3 + i] = v.f[i]

    def calc_force(x: ti.ScalarField, Ax: ti.ScalarField) -> None:
        _x(mesh, x)
        mtm.calc_force(mesh)
        _Ax(mesh, Ax)

    b: ti.ScalarField = ti.field(float, shape=(len(mesh.verts) * 3,))
    b.fill(0)
    x: ti.ScalarField = ti.field(float, shape=(len(mesh.verts) * 3,))
    rng: np.random.Generator = np.random.default_rng()
    x.from_numpy(rng.random(size=x.shape))
    ti.linalg.MatrixFreeCG(
        ti.linalg.LinearOperator(calc_force), b, x, tol=1e-6, quiet=False
    )
    np.testing.assert_allclose(x.to_numpy(), np.zeros(shape=x.shape), atol=1e-5)
