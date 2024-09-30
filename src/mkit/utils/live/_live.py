import collections
from typing import Literal, Self

import dvclive
import numpy.typing as npt


class Live:
    _live: dvclive.Live | None = None
    array: collections.defaultdict[str, list[npt.NDArray]]
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
        report: Literal["md", "notebook", "html", None] = None,
        resume: bool = False,
        save_dvc_exp: bool = True,
    ) -> None:
        self.array = collections.defaultdict(list)
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
        if self._live:
            self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._live:
            self._live.__exit__(exc_type, exc_val, exc_tb)

    def end(self) -> None:
        if self._live:
            self._live.end()

    def log_array(self, name: float | str) -> None:
        if self.enable:
            self.array[name].append(name)

    def log_metric(
        self,
        name: str,
        val: float | str,
        *,
        timestamp: bool = False,
        plot: bool = True,
    ) -> None:
        if self._live:
            self._live.log_metric(name, val, timestamp=timestamp, plot=plot)

    def next_step(self, *, milestone: bool = True) -> None:
        if self._live:
            if milestone:
                self._live.next_step()
            else:
                self._live.step += 1
