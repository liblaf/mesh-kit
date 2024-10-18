import dataclasses


@dataclasses.dataclass(kw_only=True)
class DescribeResult:
    count: int
    unique: int
    top: float
    freq: int
    mean: bool | int | float | complex
    min: bool | int | float | complex
    p25: bool | int | float | complex
    p50: bool | int | float | complex
    p75: bool | int | float | complex
    max: bool | int | float | complex
