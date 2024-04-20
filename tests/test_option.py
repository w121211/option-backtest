from datetime import datetime, timedelta
from typing import Literal

import pytest
import pandas as pd

from src.option_backtest.market import Option
from src.option_backtest.option import find_options


@pytest.fixture
def current_date():
    return pd.Timestamp("2023-01-01")


@pytest.fixture
def listed_options(current_date):
    expiry_date1 = current_date + timedelta(days=20)
    expiry_date2 = current_date + timedelta(days=40)

    options = [
        Option(
            id="c_k=100_exp=20_delta=0.6",
            option_type="call",
            strike=100,
            expiry_date=expiry_date1,
            df=pd.DataFrame({"date": [current_date], "delta": [0.6]}).set_index("date"),
        ),
        Option(
            id="p_k=100_exp=20_delta=0.4",
            option_type="put",
            strike=100,
            expiry_date=expiry_date1,
            df=pd.DataFrame({"date": [current_date], "delta": [0.4]}).set_index("date"),
        ),
        Option(
            id="c_k=200_exp=40_delta=0.4",
            option_type="call",
            strike=200,
            expiry_date=expiry_date2,
            df=pd.DataFrame({"date": [current_date], "delta": [0.4]}).set_index("date"),
        ),
        Option(
            id="p_k=200_exp=40_delta=0.6",
            option_type="put",
            strike=200,
            expiry_date=expiry_date2,
            df=pd.DataFrame({"date": [current_date], "delta": [0.6]}).set_index("date"),
        ),
    ]
    return {opt.id: opt for opt in options}


def test_find_options_expiration_gt(current_date, listed_options, snapshot):
    actual = find_options(
        listed_options,
        current_stock_price=150,
        current_date=current_date,
        finder={"option_type": "call", "expiration_gt": 30},
    )
    assert actual == snapshot(name="expiration_gt=30")


def test_find_options_delta_gt(current_date, listed_options, snapshot):
    actual0 = find_options(
        listed_options,
        current_stock_price=150,
        current_date=current_date,
        finder={"option_type": "call", "delta_gt": 0.5},
    )
    assert actual0 == snapshot(name="delta_gt=0.5")

    actual1 = find_options(
        listed_options,
        current_stock_price=150,
        current_date=current_date,
        finder={"option_type": "call", "delta_gt": 0.3},
    )
    assert actual1 == snapshot(name="delta_gt=0.3")


def test_find_options_delta_lt(current_date, listed_options, snapshot):
    actual0 = find_options(
        listed_options,
        current_stock_price=150,
        current_date=current_date,
        finder={"option_type": "call", "delta_lt": 0.5},
    )
    assert actual0 == snapshot(name="delta_lt=0.5")

    actual1 = find_options(
        listed_options,
        current_stock_price=150,
        current_date=current_date,
        finder={"option_type": "call", "delta_lt": 0.7},
    )
    assert actual1 == snapshot(name="delta_lt=0.7")


def test_find_options_sort_by_delta_near(current_date, listed_options, snapshot):
    actual = find_options(
        listed_options,
        current_stock_price=150,
        current_date=current_date,
        finder={"option_type": "call", "sort_by_delta_near": 0.45},
    )
    assert actual == snapshot(name="sort_by_delta_near=0.45")
