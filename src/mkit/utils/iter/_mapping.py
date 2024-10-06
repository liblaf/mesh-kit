from collections.abc import Mapping


def merge_mapping(origin: Mapping, update: Mapping) -> dict:
    original: dict = dict(origin)
    for key, value in update.items():
        if isinstance(value, dict) and key in original:
            if not isinstance(original[key], dict):
                msg: str = (
                    f"Config variables contradict each other: "
                    f"Key '{key}' is both a value and a nested dict."
                )
                raise ValueError(msg)
            original[key] = merge_mapping(original[key], value)
        else:
            original[key] = value
    return original
