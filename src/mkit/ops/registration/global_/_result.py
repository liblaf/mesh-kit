import dataclasses

import mkit.typing.numpy as tn


@dataclasses.dataclass(kw_only=True)
class GlobalRegistrationResult:
    correspondence_set: tn.IN2
    fitness: float
    inlier_rmse: float
    transform: tn.F44
