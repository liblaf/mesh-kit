import abc
from typing import Any

import mkit.typing.numpy as nt
from mkit.ops.registration.rigid import RigidRegistrationResult


class RigidRegistrationMethod(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        source: Any,
        target: Any,
        source_weight: nt.FNLike | None = None,
        target_weight: nt.FNLike | None = None,
    ) -> RigidRegistrationResult: ...
