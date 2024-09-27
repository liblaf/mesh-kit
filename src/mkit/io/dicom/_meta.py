from typing import Literal

from pydantic import BaseModel


class AcquisitionMeta(BaseModel):
    age: str
    birth_date: str
    date: str
    id: str
    name: str
    sex: Literal["F", "M"]
    time: str


class PatientMeta(BaseModel):
    acquisitions: list[AcquisitionMeta] = []
    birth_date: str
    id: str
    name: str
    sex: Literal["F", "M"]


class DatasetMeta(BaseModel):
    patients: dict[str, PatientMeta] = {}
