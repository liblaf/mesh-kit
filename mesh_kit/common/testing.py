from collections.abc import Sequence


def assert_shape(actual: Sequence[int], expected: Sequence[int]) -> None:
    assert len(actual) == len(expected), (actual, expected)
    for actual, expected in zip(actual, expected):
        if expected > 0:
            assert actual == expected, (actual, expected)
