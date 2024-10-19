import dataclasses

import mkit.typing.numpy as tn


@dataclasses.dataclass(kw_only=True)
class RigidRegistrationResult:
    transform: tn.F44
    cost: float
