import dataclasses

import mkit.typing.numpy as nt


@dataclasses.dataclass(kw_only=True)
class GlobalRegistrationResult:
    correspondence_set: nt.IN2
    fitness: float
    inlier_rmse: float
    transform: nt.D44
