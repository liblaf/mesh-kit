import dataclasses

import mkit.typing.numpy as nt


@dataclasses.dataclass(kw_only=True)
class NonRigidRegistrationResult:
    points: nt.FN3
