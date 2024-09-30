import abc

from mkit.ops.registration.non_rigid import NonRigidRegistrationResult


class NonRigidRegistrationMethod(abc.ABC):
    @abc.abstractmethod
    def run(self) -> NonRigidRegistrationResult: ...
