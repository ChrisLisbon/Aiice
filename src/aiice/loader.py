from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np

from aiice.constants import MAX_SPLIT_FRACTION, MIN_SPLIT_FRACTION
from aiice.core.huggingface import HfDatasetClient


class Loader:
    def __init__(self):
        """
        Dataset Loader with a Hugging Face dataset client.
        """
        self._hf = HfDatasetClient()

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Return shape of a single dataset sample.
        """
        return self._hf.shape

    @property
    def dataset_start(self) -> date:
        """
        Return earliest available date in the dataset.
        """
        return self._hf.dataset_start

    @property
    def dataset_end(self) -> date:
        """
        Return latest available date in the dataset.
        """
        return self._hf.dataset_end

    def info(self, per_year: bool = False) -> dict[str, any]:
        """
        Collect dataset statistics.

        Parameters
        ----------
        per_year : bool
            If True, include per-year statistics.
        """
        return self._hf.info(per_year=per_year)

    def download(
        self,
        local_dir: str,
        start: date | None = None,
        end: date | None = None,
        step: date | None = None,
        threads: int = 24,
    ) -> list[str | None]:
        """
        Download dataset files to a local directory in parallel.

        Parameters
        ----------
        local_dir : str
            Directory to save files.
        start : date, optional
            Start date for files.
        end : date, optional
            End date for files.
        step : int, optional
            Step in days between files.
        threads : int
            Number of parallel download threads.
        """
        filenames = self._hf.get_filenames(start=start, end=end, step=step)
        with ThreadPoolExecutor(max_workers=threads) as pool:
            return list(
                pool.map(
                    lambda filename: self._hf.download_file(
                        filename=filename, local_dir=local_dir
                    ),
                    filenames,
                )
            )

    def get(
        self,
        start: date | None = None,
        end: date | None = None,
        step: int | None = None,
        test_size: float | None = None,
        threads: int = 24,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Load dataset into memory, optionally split into train/test.

        Parameters
        ----------
        start : date, optional
            Start date for files.
        end : date, optional
            End date for files.
        step : int, optional
            Step in days between files.
        test_size : float, optional
            Fraction of data for test split (0–1).
        threads : int
            Number of parallel loading threads.
        """
        filenames = self._hf.get_filenames(start=start, end=end, step=step)
        if test_size is None:
            return self._get_files(filenames=filenames, threads=threads)

        if not MIN_SPLIT_FRACTION <= test_size <= MAX_SPLIT_FRACTION:
            raise ValueError(
                f"Test size should be between {MAX_SPLIT_FRACTION} and {MAX_SPLIT_FRACTION}"
            )

        if test_size == MIN_SPLIT_FRACTION:
            return self._get_files(filenames, threads), np.empty((0,))

        if test_size == MAX_SPLIT_FRACTION:
            return np.empty((0,)), self._get_files(filenames, threads)

        split_index = len(filenames) - int(len(filenames) * test_size)
        train_files = filenames[:split_index]
        test_files = filenames[split_index:]

        return (
            self._get_files(train_files, threads),
            self._get_files(test_files, threads),
        )

    def _get_file(self, filename: str) -> np.ndarray:
        npy = self._hf.read_file(filename=filename)
        if npy is None:
            raise ValueError(f"Remote file {filename} not found")
        return npy

    def _get_files(self, filenames: list[str], threads: int):
        with ThreadPoolExecutor(max_workers=threads) as pool:
            npys = list(pool.map(lambda filename: self._get_file(filename), filenames))
        return np.array(npys)
