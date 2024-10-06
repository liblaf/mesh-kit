from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard

import mkit

if TYPE_CHECKING:
    import jax


def is_jax(obj: Any) -> TypeGuard[jax.Array]:
    return mkit.typing.is_instance_named_partial(obj, "jax.Array")
