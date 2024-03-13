import pytest
import taichi as ti


@pytest.fixture(autouse=True)
def ti_init() -> None:
    ti.init(default_fp=ti.f64)
