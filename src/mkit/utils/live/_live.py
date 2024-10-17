import collections
from types import TracebackType
from typing import Any, Literal, Self

import dvclive
import numpy.typing as npt
from loguru import logger

import mkit


class Live:
    _live: dvclive.Live | None = None
    records: dict[int, dict[str, Any]]
    enable: bool = True

    def __init__(
        self,
        *,
        cache_images: bool = False,
        dir: str = "dvclive",  # noqa: A002
        dvcyaml: str | None = "dvc.yaml",
        enable: bool = True,
        exp_message: str | None = None,
        exp_name: str | None = None,
        monitor_system: bool = False,
        report: Literal["md", "notebook", "html"] | None = None,
        resume: bool = False,
        save_dvc_exp: bool = False,
    ) -> None:
        self.records = collections.defaultdict(dict)
        self.enable = enable
        if enable:
            self._live = dvclive.Live(
                dir=dir,
                resume=resume,
                report=report,
                save_dvc_exp=save_dvc_exp,
                dvcyaml=dvcyaml,
                cache_images=cache_images,
                exp_name=exp_name,
                exp_message=exp_message,
                monitor_system=monitor_system,
            )
        else:
            self._live = None

    def __enter__(self) -> Self:
        if self._live is None:
            return self
        self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._live is None:
            return
        self._live.__exit__(exc_type, exc_val, exc_tb)

    def end(self) -> None:
        if self._live is None:
            return
        self._live.end()

    def log_array(self, name: str, array: npt.ArrayLike) -> None:
        if not self.enable:
            return
        self.records[self.step][name] = mkit.math.as_numpy(array)

    def log_metric(
        self,
        name: str,
        val: Any,
        *,
        timestamp: bool = False,
        plot: bool = True,
    ) -> None:
        if self._live is None:
            return
        val: int | float | str = mkit.math.as_scalar(val)
        self.records[self.step][name] = val
        self._live.log_metric(name, val, timestamp=timestamp, plot=plot)

    def next_step(self, *, milestone: bool = True) -> None:
        if self._live is None:
            return
        if milestone:
            metrics: str = ""
            for k, v in self.records[self.step].items():
                metrics += f"{k}: {v}\n"
            self._live.make_report()
            logger.info("step {}: {}", self.step, metrics)
            self._live.next_step()
        else:
            self._live.step += 1

    @property
    def step(self) -> int:
        if self._live:
            return self._live.step
        return 0

    @step.setter
    def step(self, step: int) -> None:
        if self._live:
            self._live.step = step
