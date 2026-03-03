import functools
import time
from datetime import date
from pathlib import Path

import httpx

from aiice.constants import DEFAULT_BACKOFF, DEFAULT_RETRIES

RETRY_EXCEPTIONS = (
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.TimeoutException,
)


def retry_on_network_errors(
    retries: int = DEFAULT_RETRIES, backoff: float = DEFAULT_BACKOFF
):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, retries + 1):
                try:
                    return func(*args, **kwargs)
                except RETRY_EXCEPTIONS as e:
                    if attempt < retries:
                        time.sleep(backoff * attempt)
                    else:
                        raise e

        return wrapper

    return decorator


def get_filename_template(d: date) -> str:
    # filename looks like: global_series/1999/osisaf_19991101.npy
    return f"global_series/{d.year}/osisaf_{d.year}{d.month:02d}{d.day:02d}.npy"


def get_date_from_filename_template(f: str) -> date:
    # filename looks like: global_series/1999/osisaf_19991101.npy
    name = Path(f).name
    date_part = name.removeprefix("osisaf_").removesuffix(".npy")

    year = int(date_part[0:4])
    month = int(date_part[4:6])
    day = int(date_part[6:8])
    return date(year, month, day)
