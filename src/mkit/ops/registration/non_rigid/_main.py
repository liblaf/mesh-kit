import mkit.ops.registration.non_rigid as nr


def non_rigid_registration(
    method: nr.NonRigidRegistrationMethod,
) -> nr.NonRigidRegistrationResult:
    result: nr.NonRigidRegistrationResult = method.run()
    return result
