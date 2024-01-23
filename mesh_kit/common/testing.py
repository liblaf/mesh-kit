from collections.abc import Sequence


def assert_shape(actual: Sequence[int], expected: Sequence[int]) -> None:
    assert len(actual) == len(expected), (actual, expected)
    for a, e in zip(actual, expected):
        if e > 0:
            assert a == e, (actual, expected)
