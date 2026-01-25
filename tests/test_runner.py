import math

import torch
from torch.utils.data import DataLoader

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

    dataloader = DataLoader(dataset=dataset, batch_size=2)
    model = DummyModel()
    runner = Runner(model=model, dataloader=dataloader, metrics=["mae", "mse"])

    report = runner.run()

    for metric in ["mae", "mse"]:
        assert report[metric]["count"] == len(dataloader)
        assert not math.isnan(report[metric]["mean"])
        assert not math.isnan(report[metric]["last"])
        assert report[metric]["mean"] >= 0
        assert report[metric]["last"] >= 0
