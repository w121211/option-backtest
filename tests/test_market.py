from datetime import datetime, timedelta
import pytest
import pandas as pd

from src.option_backtest.market import (
    Stock,
    Option,
    Account,
    Trade,
    after_market,
    execute_trades,
)


@pytest.fixture
def current_date():
    return pd.Timestamp("2023-01-01")


@pytest.fixture
def listed_stock(current_date):
    dates = pd.date_range(start=current_date, periods=2, freq="D")
    df = pd.DataFrame({"price": [100, 105]}, index=dates)
    return Stock(id="XYZ", df=df)


@pytest.fixture
def listed_options(current_date):
    expiry_date1 = current_date + timedelta(days=10)
    expiry_date2 = current_date - timedelta(days=1)
    expiry_date3 = current_date

    options = [
        Option(
            id="XYZ_C_110_10",
            option_type="call",
            strike=110,
            expiry_date=expiry_date1,
            df=pd.DataFrame(
                {"price": [5, 6]},
                index=pd.date_range(start=current_date, periods=2, freq="D"),
            ),
        ),
        Option(
            id="XYZ_P_90_m1",
            option_type="put",
            strike=90,
            expiry_date=expiry_date2,
            df=pd.DataFrame(
                {"price": [3, 4]},
                index=pd.date_range(start=current_date, periods=2, freq="D"),
            ),
        ),
        Option(
            id="XYZ_C_90_0",
            option_type="call",
            strike=90,
            expiry_date=expiry_date3,
            df=pd.DataFrame(
                {"price": [3, 4]},
                index=pd.date_range(start=current_date, periods=2, freq="D"),
            ),
        ),
    ]
    return {opt.id: opt for opt in options}


def test_execute_trades(current_date, listed_stock, listed_options, snapshot):
    _options = list(listed_options.values())
    account = Account(cash=1000, stock=listed_stock, options={})

    trades = [
        Trade(
            asset_type="option",
            asset_id=_options[0].id,
            position=2,
            date=current_date,
        ),
        Trade(
            asset_type="option",
            asset_id=_options[1].id,
            position=-1,
            date=current_date,
        ),
    ]
    account1 = execute_trades(listed_options, account, current_date, trades)
    assert account1 == snapshot

    trades = [
        Trade(asset_type="stock", asset_id="XYZ", position=10, date=current_date),
    ]
    account2 = execute_trades(listed_options, account1, current_date, trades)
    assert account2 == snapshot


def test_after_market_expired_call_exercised(
    current_date, listed_stock, listed_options, snapshot
):
    account = Account(cash=1000, stock=listed_stock, options=listed_options)
    account.options["XYZ_C_110_10"].position = 1
    account.options["XYZ_P_90_m1"].position = 1
    account.options["XYZ_C_90_0"].position = 1

    listed_options, updated_account = after_market(
        current_date, listed_options, account
    )

    # assert len(available_options) == 1
    # assert "XYZ_C_110_20230112" in available_options
    # assert "XYZ_P_90_20230101" not in available_options

    # assert len(updated_account.options) == 1
    # assert "XYZ_C_110_20230112" in updated_account.options
    # assert "XYZ_P_90_20230101" not in updated_account.options

    # assert updated_account.stock.position == 0
    # assert updated_account.cash == 1000

    assert listed_options == snapshot
    assert updated_account == snapshot


# def test_after_market_expired_put_exercised(current_date, listed_stock, listed_options):
#     listed_stock.df.at[current_date, "price"] = 80
#     account = Account(cash=1000, stock=listed_stock, options=listed_options)
#     account.stock.position = 1
#     account.options["XYZ_C_110_20230112"].position = 1
#     account.options["XYZ_P_90_20230101"].position = 1

#     available_options, updated_account = after_market(
#         current_date, listed_options, account
#     )

#     assert len(available_options) == 1
#     assert "XYZ_C_110_20230112" in available_options
#     assert "XYZ_P_90_20230101" not in available_options

#     assert len(updated_account.options) == 1
#     assert "XYZ_C_110_20230112" in updated_account.options
#     assert "XYZ_P_90_20230101" not in updated_account.options

#     assert updated_account.stock.position == 0
#     assert updated_account.cash == 1090
