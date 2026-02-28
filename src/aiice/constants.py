from datetime import date

# hugging face constants

HF_BASE_URL: str = "https://huggingface.co"
HF_REPO_TYPE: str = "dataset"
HF_DATASET_REPO: str = "ITMO-NSS/Aiice"
HF_PACKAGE_NAME: str = "aiice"

# dataset constants

MIN_DATASET_START: date = date(1980, 1, 1)
MAX_DATASET_END: date = date(2025, 7, 1)
DATASET_SHAPE: tuple[int, int] = (432, 432)

BYTES_IN_MB: int = 1024 * 1024

YEAR_STATS_CACHE_SIZE: int = 64

KEY_DATASET_START: str = "start_date"
KEY_DATASET_END: str = "end_date"
KEY_PER_YEAR: str = "per_year"
KEY_SHAPE: str = "shape"
KEY_FILES: str = "files"
KEY_SIZE_BYTES: str = "size_bytes"
KEY_SIZE_MB: str = "size_mb"

DEFAULT_RETRIES: int = 5
DEFAULT_BACKOFF: float = 3.0

# metrics constants

DEFAULT_SSIM_KERNEL_WINDOW_SIZE: int = 11

MAE_METRIC: str = "mae"
MSE_METRIC: str = "mse"
RMSE_METRIC: str = "rmse"
PSNR_METRIC: str = "psnr"
BIN_ACCURACY_METRIC: str = "bin_accuracy"
SSIM_METRIC: str = "ssim"

MEAN_STAT: str = "mean"
LAST_STAT: str = "last"
COUNT_STAT: str = "count"
MIN_STAT: str = "min"
MAX_STAT: str = "max"
