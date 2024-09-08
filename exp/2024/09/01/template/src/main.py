from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import pyvista as pv
import trimesh as tm
from icecream import ic

import mkit
import mkit.typing as t
import mkit.typing.numpy as n
from mkit.ext import ICTFaceKit


class Config(mkit.cli.CLIBaseConfig):
    pass


def main(cfg: Config) -> None:
    ict: ICTFaceKit = ICTFaceKit()
    ict_face: pv.PolyData = ict.narrow_face
    pcd: pv.PolyData = ict_face.cast_to_poly_points()
    ic(pcd)
    ic(pcd.regular_faces)
    ic(tm.Trimesh(pcd.points, pcd.regular_faces, process=True))

    # ict_face.point_data["normal"] = ict_face.point_normals
    # ict_face.save("data/ict.vtp")
    # sample: pv.PolyData = sample_surface(ict_face, 500, point_data=ict_face.point_data)
    # sample.save("data/sample.vtp")


def icp(
    _source: t.AnySurfaceMesh,
    _target: t.AnySurfaceMesh,
    initial: npt.ArrayLike | None = None,
    *,
    max_iter: int = 100,
    reflection: bool = False,
    inverse: bool = False,
    samples: int = 10000,
    scale: bool = True,
    threshold: float = 1e-6,
    translation: bool = True,
    weights: npt.ArrayLike | None = None,
) -> tuple[n.D44, float]:
    matrix: n.D44
    cost: float
    if inverse:
        matrix, cost = icp(
            _target,
            _source,
            tm.transformations.inverse_matrix(initial),
            max_iter=max_iter,
            reflection=reflection,
            samples=samples,
            scale=scale,
            threshold=threshold,
            translation=translation,
            weights=weights,
        )
        matrix = tm.transformations.inverse_matrix(matrix)
        return matrix, cost
    if weights is not None:
        raise NotImplementedError
    source: tm.Trimesh = mkit.io.as_trimesh(_source)
    target: tm.Trimesh = mkit.io.as_trimesh(_target)
    if initial is not None:
        source = source.apply_transform(initial)
    source_denorm_trans: n.D44 = mkit.ops.transform.denormalize_transform(source)
    target_denorm_trans: n.D44 = mkit.ops.transform.denormalize_transform(target)
    source_norm_trans: n.D44 = mkit.ops.transform.normalize_transform(source)
    target_norm_trans: n.D44 = mkit.ops.transform.normalize_transform(target)
    source = source.apply_transform(source_norm_trans)
    target = target.apply_transform(target_norm_trans)
    source_pcd: pv.PolyData = sample_surface(source, samples)
    target_pcd: pv.PolyData = sample_surface(target, samples)
    matrix, _, cost = tm.registration.icp(
        source_pcd,
        target_pcd,
        threshold=threshold,
        max_iterations=max_iter,
        reflection=reflection,
        translation=translation,
        scale=scale,
    )
    matrix = tm.transformations.concatenate_matrices(
        target_denorm_trans, matrix, source_norm_trans, initial
    )
    return matrix, cost


def _icp(
    _source: Any,
    _target: Any,
    *,
    max_iter: int = 100,
    reflection: bool = False,
    scale: bool = True,
    threshold: float = 1e-6,
    translation: bool = True,
    weights: npt.ArrayLike | None = None,
    target_is_surface: bool = False,
) -> tuple[n.D44, float]:
    source: pv.PolyData = mkit.io.as_polydata(_source)
    target: tm.Trimesh = mkit.io.as_trimesh(_target)
    for _ in range(max_iter):
        closest: n.DN3
        if target_is_surface:
            closest, _, _ = target.nearest.on_surface(source.points)
        else:
            _, vertex_id = target.nearest.vertex(source.points)
            closest = target.vertices[vertex_id]
        matrix, _, cost = tm.registration.procrustes(
            source.points,
            closest,
            reflection=reflection,
            scale=scale,
            translation=translation,
        )


def sample_surface(
    _mesh: t.AnySurfaceMesh,
    count: int,
    point_data: Mapping[str, npt.ArrayLike] | pv.DataSetAttributes | None = None,
) -> pv.PolyData:
    mesh: tm.Trimesh = mkit.io.as_trimesh(_mesh)
    samples: n.DN3
    face_index: n.IN
    samples, face_index = mesh.sample(count, return_index=True)
    sample: pv.PolyData = pv.wrap(samples)
    if point_data is not None:
        barycentric: n.DN3 = tm.triangles.points_to_barycentric(
            mesh.triangles[face_index], samples
        )
        for k, v in point_data.items():
            data: npt.NDArray = np.asarray(v)[mesh.faces[face_index]]
            data_interp: npt.NDArray = np.einsum("ij,ij...->i...", barycentric, data)
            sample[k] = mkit.math.numpy.cast(data_interp, data.dtype)
    return sample


if __name__ == "__main__":
    mkit.cli.run(main)
