import math

import numpy as np
import torch

from aiice.preprocess import SlidingWindowDataset
from aiice.runner import Runner


class DummyModel(torch.nn.Module):
    def forward(self, x):
        return x[-1:] + 1.0


def test_runner_dummy():
    data = torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(5, 2, 3)

    dataset = SlidingWindowDataset(
        data=data,
        pre_history_len=2,
        forecast_len=1,
    )

    assert len(dataset) == 5 - 2 - 1 + 1  # =3

    model = DummyModel()
    runner = Runner(model=model, dataset=dataset, metrics=["mae", "mse"])

    report = runner.run()

    for metric in ["mae", "mse"]:
        assert report[metric]["count"] == len(dataset)
        assert not math.isnan(report[metric]["mean"])
        assert not math.isnan(report[metric]["last"])
        assert report[metric]["mean"] >= 0
        assert report[metric]["last"] >= 0
