from collections.abc import Mapping


def merge_mapping(origin: Mapping, update: Mapping) -> dict:
    """Updates the original dict with the new data. Similar to `dict.update()`, but works with nested dicts.

    References:
        1. [ConfZ/confz/loaders/loader.py:L10-L28)(https://github.com/Zuehlke/ConfZ/blob/6c99cc2a2938e231590dceeef66749ccf2eb6b4c/confz/loaders/loader.py#L10-L28)
    """
    original: dict = dict(origin)
    for key, value in update.items():
        if isinstance(value, dict) and key in original:
            if not isinstance(original[key], dict):
                msg: str = (
                    "Config variables contradict each other: "
                    f"Key {key:!r} is both a value and a nested dict."
                )
                raise ValueError(msg)
            original[key] = merge_mapping(original[key], value)
        else:
            original[key] = value
    return original
