import dataclasses

import mkit.typing.numpy as tn


@dataclasses.dataclass(kw_only=True)
class NonRigidRegistrationResult:
    points: tn.FN3
