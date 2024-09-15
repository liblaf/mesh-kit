from typing import Literal

from pydantic import BaseModel


class MetaAcquisition(BaseModel):
    age: str
    birth_date: str
    date: str
    id: str
    name: str
    sex: Literal["F", "M"]
    time: str


class MetaPatient(BaseModel):
    acquisitions: list[MetaAcquisition] = []
    birth_date: str
    id: str
    name: str
    sex: Literal["F", "M"]


class MetaDataset(BaseModel):
    patients: dict[str, MetaPatient] = {}
