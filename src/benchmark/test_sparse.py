from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest
import sparse
from mkit.sparse import sparse_mask
from pytest_benchmark.fixture import BenchmarkFixture

if TYPE_CHECKING:
    from numpy.random import Generator


def sparse_mask_naive(
    arr: sparse.COO, masks: tuple[npt.NDArray[np.bool], ...]
) -> sparse.COO:
    for i, mask in enumerate(masks):
        idx: list = [slice(None) for _ in masks]
        idx[i] = mask
        arr = arr[tuple(idx)]
    return arr


@pytest.fixture()
def arr() -> sparse.COO:
    return sparse.random((1000, 1000))  # pyright: ignore [reportReturnType]


@pytest.fixture()
def masks() -> tuple[npt.NDArray[np.bool], ...]:
    generator: Generator = np.random.default_rng()
    mask: npt.NDArray[np.bool] = generator.choice([False, True], (1000,))
    return mask, mask


def test_sparse_mask(
    benchmark: BenchmarkFixture,
    arr: sparse.COO,
    masks: tuple[npt.NDArray[np.bool], ...],
) -> None:
    assert sparse.all(sparse_mask(arr, masks) == sparse_mask_naive(arr, masks))
    benchmark(sparse_mask, arr, masks)


def test_sparse_mask_naive(
    benchmark: BenchmarkFixture,
    arr: sparse.COO,
    masks: tuple[npt.NDArray[np.bool], ...],
) -> None:
    benchmark(sparse_mask_naive, arr, masks)
