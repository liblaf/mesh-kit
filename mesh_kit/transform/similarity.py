import pytorch3d.ops.points_alignment
from pytorch3d.ops.points_alignment import SimilarityTransform
from torch import Tensor


def apply(X: Tensor, transform: SimilarityTransform) -> Tensor:
    assert len(X.shape) == 3
    return pytorch3d.ops.points_alignment._apply_similarity_transform(
        X=X, R=transform.R, T=transform.T, s=transform.s
    )
