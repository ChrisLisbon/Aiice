import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from aiice.metrics import Evaluator, MetricFn


class Runner:
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        metrics: dict[str, MetricFn] | list[str] | None = None,
        device: str | None = None,
    ):
        self._model = model
        self._data = dataloader
        self._device = device
        self._evaluator = Evaluator(metrics=metrics, accumulate=True)
        self._last_report: dict[str, list[float]] = {
            k: [] for k in self._evaluator._metrics
        }

    @property
    def last_report(self):
        return self._last_report

    def run(self) -> dict[str, list[float]]:
        for x, y in tqdm(self._data):
            x, y = x.to(self._device), y.to(self._device)
            pred = self._model(x)
            self._evaluator.eval(y, pred)

        self._last_report = self._evaluator.report()
        return self.last_report
