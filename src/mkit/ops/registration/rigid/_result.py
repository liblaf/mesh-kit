import dataclasses

import mkit.typing.numpy as nt


@dataclasses.dataclass(kw_only=True)
class RigidRegistrationResult:
    transform: nt.D44
    cost: float
