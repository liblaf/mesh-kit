import functools
import inspect
from collections.abc import Callable
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")


def kwargs_to_positional(func: Callable[_P, _T]) -> Callable[_P, _T]:
    """Convert **kwargs arguments to positional.

    Reference:
        1. <https://stackoverflow.com/a/49836730/18410348>
    """
    sig: inspect.Signature = inspect.signature(func)
    if any(v.kind == inspect.Parameter.VAR_KEYWORD for v in sig.parameters.values()):
        msg: str = f"Arbitrary keyword arguments are not supported by {__name__}"
        raise TypeError(msg)

    @functools.wraps(func)
    def wrapped_func(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        ba: inspect.BoundArguments = sig.bind(*args, **kwargs)
        ba.apply_defaults()
        assert len(ba.kwargs) == 0
        return func(*ba.args, **ba.kwargs)

    return wrapped_func
