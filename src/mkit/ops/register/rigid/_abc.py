from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import attrs
import numpy as np

import mkit.math as mm
import mkit.ops as mo
import mkit.typing.numpy as tn
import mkit.utils as mu

if TYPE_CHECKING:
    from mkit.ops.register import rigid


@attrs.define
class RigidRegistrationBase(abc.ABC):
    source: Any
    target: Any
    source_weights: tn.FN | None = attrs.field(
        default=None, converter=lambda x: None if x is None else mm.as_numpy(x)
    )
    target_weights: tn.FN | None = attrs.field(
        default=None, converter=lambda x: None if x is None else mm.as_numpy(x)
    )

    # registration params
    inverse: bool = False
    reflection: bool = True
    scale: bool = True
    translation: bool = True

    # preprocess params
    init_transform: tn.F44 = attrs.field(default=np.eye(4), converter=mm.as_numpy)
    normalize: bool = True
    simplify: bool = True

    # auxiliary variables
    source_normalization_transformation: tn.F44 | None = attrs.field(
        default=None, init=False
    )
    target_denormalization_transformation: tn.F44 | None = attrs.field(
        default=None, init=False
    )

    def preprocess_source(self) -> Any:
        mesh: Any = self.source
        if self.simplify:
            mesh = mo.simplify(mesh)
        mesh = mo.transform(mesh, self.init_transform)
        if self.normalize:
            self.source_normalization_transformation = mo.normalization_transformation(
                mesh
            )
            mesh = mo.normalize(mesh)
        return mesh

    def preprocess_target(self) -> Any:
        mesh: Any = self.target
        if self.simplify:
            mesh = mo.simplify(mesh)
        if self.normalize:
            self.target_denormalization_transformation = (
                mo.denormalization_transformation(mesh)
            )
            mesh = mo.normalize(mesh)
        return mesh

    def postprocess(
        self, result: rigid.RigidRegistrationResult
    ) -> rigid.RigidRegistrationResult:
        result.transformation = mo.concatenate_matrices(
            self.target_denormalization_transformation,
            result.transformation,
            self.source_normalization_transformation,
            self.init_transform,
        )
        return result

    @mu.log_time
    def register(self) -> rigid.RigidRegistrationResult:
        source: Any = self.preprocess_source()
        target: Any = self.preprocess_target()
        source_weights: tn.FN | None = self.source_weights
        target_weights: tn.FN | None = self.target_weights
        if self.inverse:
            source, target = target, source
            source_weights, target_weights = target_weights, source_weights
        result: rigid.RigidRegistrationResult = self._register(
            source, target, source_weights=source_weights, target_weights=target_weights
        )
        if self.inverse:
            result.transformation = np.linalg.inv(result.transformation)
        return self.postprocess(result)

    @abc.abstractmethod
    def _register(
        self,
        source: Any,
        target: Any,
        *,
        source_weights: tn.FNLike | None = None,
        target_weights: tn.FNLike | None = None,
    ) -> rigid.RigidRegistrationResult: ...
