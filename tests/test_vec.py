from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.option_backtest.backtest_vec import (
    calculate_portfolio_values,
    rollover_short_atm_calls,
    rollovers_to_signals,
    Option,
)


def test_rollover_short_atm_calls(snapshot):
    # Define test input parameters
    available_options = [
        Option(
            id="XYZ230630C00050000",
            option_type="call",
            strike=50,
            expiry_date=datetime(2023, 6, 30),
        ),
        Option(
            id="XYZ230630C00060000",
            option_type="call",
            strike=60,
            expiry_date=datetime(2023, 6, 30),
        ),
        Option(
            id="XYZ230630C00070000",
            option_type="call",
            strike=70,
            expiry_date=datetime(2023, 6, 30),
        ),
        Option(
            id="XYZ230930C00050000",
            option_type="call",
            strike=50,
            expiry_date=datetime(2023, 9, 30),
        ),
        Option(
            id="XYZ230930C00060000",
            option_type="call",
            strike=60,
            expiry_date=datetime(2023, 9, 30),
        ),
        Option(
            id="XYZ230930C00070000",
            option_type="call",
            strike=70,
            expiry_date=datetime(2023, 9, 30),
        ),
    ]
    prices_df = pd.DataFrame(
        [
            {"date": "2023-01-01", "price": 55},
            {"date": "2023-02-01", "price": 58},
            {"date": "2023-03-01", "price": 62},
            {"date": "2023-04-01", "price": 60},
            {"date": "2023-05-01", "price": 57},
            {"date": "2023-06-01", "price": 59},
            {"date": "2023-07-01", "price": 63},
            {"date": "2023-08-01", "price": 61},
            {"date": "2023-09-01", "price": 56},
        ]
    )

    # Call the function to rollover short ATM calls
    options = rollover_short_atm_calls(available_options, prices_df)

    assert options == snapshot


def test_rollovers_to_signals():
    # Create sample rollovers data
    rollovers = [
        Option(
            id="XYZ230630C00100000",
            option_type="call",
            strike=100.0,
            expiry_date=datetime(2023, 6, 30),
            position=1,
            entry_date=datetime(2023, 6, 1),
            exit_date=datetime(2023, 6, 20),
        ),
        Option(
            id="XYZ230731C00105000",
            option_type="call",
            strike=105.0,
            expiry_date=datetime(2023, 7, 31),
            position=-1,
            entry_date=datetime(2023, 6, 21),
            exit_date=datetime(2023, 7, 15),
        ),
        Option(
            id="XYZ230831C00110000",
            option_type="call",
            strike=110.0,
            expiry_date=datetime(2023, 8, 31),
            position=1,
            entry_date=datetime(2023, 7, 16),
            exit_date=None,
        ),
    ]

    start_date = datetime(2023, 6, 1)
    end_date = datetime(2023, 8, 31)

    # Call the rollovers_to_signals function
    signals_df = rollovers_to_signals(rollovers, start_date, end_date)

    # Check the expected output
    expected_columns = [
        "XYZ230630C00100000",
        "XYZ230731C00105000",
        "XYZ230831C00110000",
    ]
    assert list(signals_df.columns) == expected_columns

    assert signals_df.loc[datetime(2023, 6, 1), "XYZ230630C00100000"] == 1
    assert signals_df.loc[datetime(2023, 6, 20), "XYZ230630C00100000"] == -1
    assert signals_df.loc[datetime(2023, 6, 21), "XYZ230731C00105000"] == -1
    assert signals_df.loc[datetime(2023, 7, 15), "XYZ230731C00105000"] == 1
    assert signals_df.loc[datetime(2023, 7, 16), "XYZ230831C00110000"] == 1


def test_rollovers_to_signals_invalid_position():
    # Check that an exception is raised when position is None
    invalid_rollover = Option(
        id="XYZ230930C00115000",
        option_type="call",
        strike=115.0,
        expiry_date=datetime(2023, 9, 30),
        position=None,
        entry_date=datetime(2023, 8, 1),
        exit_date=None,
    )
    start_date = datetime(2023, 6, 1)
    end_date = datetime(2023, 9, 30)

    with pytest.raises(Exception) as excinfo:
        rollovers_to_signals([invalid_rollover], start_date, end_date)
    assert "Option XYZ230930C00115000 has a None position." in str(excinfo.value)


def test_calculate_portfolio_values(snapshot):
    # Create sample inputs
    positions_df = pd.DataFrame(
        {
            "AAA": [0, 1, 2, 2, 3],
            "BBB": [0, -1, 1, -2, 0],
        },
        index=pd.date_range("2023-01-01", periods=5),
    )
    market_data = {
        "AAA": pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5),
                "price": [10, 11, 12, 13, 14],
            }
        ),
        "BBB": pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5),
                "price": [20, 21, 22, 23, 24],
            }
        ),
    }

    # Call the function with sample data
    portfolio_df = calculate_portfolio_values(positions_df, market_data, initial_cash=0)

    # Assert the expected portfolio values
    # expected_portfolio_values = [110, 142, 176, 211, 247]
    # assert portfolio_df["portfolio_value"].tolist() == expected_portfolio_values

    assert portfolio_df.to_string() == snapshot
