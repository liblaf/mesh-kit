from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest
import sparse
from mkit.math.sparse import mask as sparse_mask

if TYPE_CHECKING:
    from numpy.random import Generator


def sparse_mask_naive(
    arr: sparse.COO, masks: Sequence[npt.NDArray[np.bool]]
) -> sparse.COO:
    for i, mask in enumerate(masks):
        idx: list = [slice(None) for _ in masks]
        idx[i] = mask
        arr = arr[tuple(idx)]
    return arr


@pytest.fixture
def arr() -> sparse.COO:
    return sparse.random((1000, 1000))  # pyright: ignore [reportReturnType]


@pytest.fixture
def masks(arr: sparse.COO) -> list[npt.NDArray[np.bool]]:
    generator: Generator = np.random.default_rng()
    masks: list[npt.NDArray[np.bool]] = []
    for d in arr.shape:
        mask: npt.NDArray[np.bool] = generator.choice([False, True], (d,))
        masks.append(mask)
    return masks


def test_sparse_mask(arr: sparse.COO, masks: Sequence[npt.NDArray[np.bool]]) -> None:
    assert sparse.all(sparse_mask(arr, masks) == sparse_mask_naive(arr, masks))
