import numpy as np
import trimesh
from numpy import typing as npt


def icp(
    source: trimesh.Trimesh,
    target: trimesh.Trimesh,
    *,
    inverse: bool = False,
    samples: int = 10000,
    # ICP
    initial: npt.ArrayLike | None = None,
    threshold: float = 1e-5,
    max_iterations: int = 100,
    # procrustes
    reflection: bool = True,
    translation: bool = True,
    scale: bool = True,
) -> tuple[npt.NDArray[np.float64], float]:
    """
    Args:
        source:
        target:
        inverse:
        samples: Number of samples from mesh surface to align.
        initial: (4, 4) float: Initial transformation.
        threshold: Stop when change in cost is less than threshold.
        max_iterations: Maximum number of iterations.
        reflection: If the transformation is allowed reflections.
        translation: If the transformation is allowed translations.
        scale: If the transformation is allowed scaling.

    Returns:
        source_to_target: (4, 4) float: Transform to align mesh to the other object.
        cost: Average squared distance per point.
    """

    def key_points(mesh: trimesh.Trimesh, samples: int) -> npt.NDArray[np.float64]:
        """Return a combination of mesh vertices and surface samples with vertices
        chosen by likelihood to be important to registration."""
        points: npt.NDArray[np.float64]
        if len(mesh.vertices) < (samples / 2):
            points = np.vstack(
                (mesh.vertices, mesh.sample(samples - len(mesh.vertices)))
            )
        else:
            points = mesh.sample(samples)
        return points

    source_to_target: npt.NDArray[np.float64]
    transformed: npt.NDArray[np.float64]
    cost: float
    if inverse:
        source, target = target, source
        initial = trimesh.transformations.inverse_matrix(initial)
    source_to_target, transformed, cost = trimesh.registration.icp(
        key_points(source, samples),
        key_points(target, samples),
        initial=initial,
        threshold=threshold,
        max_iterations=max_iterations,
        reflection=reflection,
        translation=translation,
        scale=scale,
    )
    if inverse:
        source_to_target = trimesh.transformations.inverse_matrix(source_to_target)
    return source_to_target, cost
