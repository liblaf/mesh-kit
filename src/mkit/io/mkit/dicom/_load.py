from __future__ import annotations

from typing import TYPE_CHECKING

import mkit.struct as ms

if TYPE_CHECKING:
    import mkit.typing as mt


def load_dicom(path: mt.StrPath) -> ms.DICOM:
    return ms.DICOM(path)
