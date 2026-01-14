import torch.nn as nn
from tqdm import tqdm

from aiice.metrics import Evaluator, MetricFn
from aiice.preprocess import SlidingWindowDataset


class Runner:
    def __init__(
        self,
        model: nn.Module,
        dataset: SlidingWindowDataset,
        metrics: dict[str, MetricFn] | list[str] | None = None,
    ):
        self._model = model
        self._dataset = dataset
        self._evaluator = Evaluator(metrics=metrics)
        self._last_report: dict[str, list[float]] = {
            k: [] for k in self._evaluator._metrics
        }

    @property
    def last_report(self):
        return self._last_report

    def run(self) -> dict[str, list[float]]:
        for x, y in tqdm(self._dataset):
            pred = self._model(x)
            self._evaluator.eval(y, pred)

        self._last_report = self._evaluator.report()
        return self.last_report
