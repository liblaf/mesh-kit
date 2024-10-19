import abc
from typing import Any

import mkit.typing.numpy as tn
from mkit.ops.registration.rigid import RigidRegistrationResult


class RigidRegistrationMethod(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        source: Any,
        target: Any,
        source_weight: tn.FNLike | None = None,
        target_weight: tn.FNLike | None = None,
    ) -> RigidRegistrationResult: ...
