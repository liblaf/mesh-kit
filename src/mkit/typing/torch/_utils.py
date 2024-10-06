from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard

import mkit

if TYPE_CHECKING:
    import torch


def is_torch(obj: Any) -> TypeGuard[torch.Tensor]:
    return mkit.typing.is_instance_named_partial(obj, "torch.Tensor")
