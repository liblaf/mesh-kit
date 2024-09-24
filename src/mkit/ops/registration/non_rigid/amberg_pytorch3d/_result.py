import dataclasses

import mkit.typing.numpy as nt


@dataclasses.dataclass(kw_only=True)
class NonRigidRegistrationResult:
    result: nt.FN3
