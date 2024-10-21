from __future__ import annotations

from typing import TYPE_CHECKING

import attrs

if TYPE_CHECKING:
    import mkit.typing.numpy as tn


@attrs.define
class RigidRegistrationResult:
    loss: float
    transformation: tn.F44
