from datetime import date
from unittest.mock import Mock

import httpx
import pytest

from aiice.core.utils import (
    get_date_from_filename_template,
    get_filename_template,
    retry_on_network_errors,
)


@pytest.fixture
def retry_decorator():
    return retry_on_network_errors(retries=3, backoff=0)


class TestRetryOnNetworkErrors:
    def test_call_n_times_on_error(self, retry_decorator):
        mock_func = Mock(side_effect=httpx.RemoteProtocolError("fail"))
        decorated = retry_decorator(mock_func)

        with pytest.raises(httpx.RemoteProtocolError):
            decorated()

        assert mock_func.call_count == 3

    def test_stop_on_success(self, retry_decorator):
        state = {"count": 0}

        def func():
            if state["count"] < 2:
                state["count"] += 1
                raise httpx.ConnectError("fail")
            return "ok"

        decorated = retry_decorator(func)
        result = decorated()
        assert result == "ok"
        assert state["count"] == 2


class TestHfDatasetClient_get_filename_template:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (date(2021, 6, 1), "global_series/2021/osisaf_20210601.npy"),
            (date(1991, 12, 12), "global_series/1991/osisaf_19911212.npy"),
        ],
    )
    def test_ok(self, value, expected):
        assert get_filename_template(value) == expected


class TestHfDatasetClient_get_date_from_filename_template:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("global_series/2021/osisaf_20210601.npy", date(2021, 6, 1)),
            ("global_series/1991/osisaf_19911212.npy", date(1991, 12, 12)),
        ],
    )
    def test_ok(self, value, expected):
        assert get_date_from_filename_template(value) == expected
