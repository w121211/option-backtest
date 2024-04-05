import os
import pprint
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm


# ------------------------------------------------------------
# Generate Stock Prices by Gometrical Brownian Motion
# ------------------------------------------------------------


def daily_returns(prices: np.ndarray) -> np.ndarray:
    daily_returns = np.diff(prices) / prices[:-1]
    return daily_returns


def prices_sigma(prices: np.ndarray) -> float:
    returns = daily_returns(prices)
    sigma = np.std(returns) * np.sqrt(252)  # Annualized volatility
    return sigma


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def call_prices(
    prices: np.ndarray,
    strikes: List[float],
    days_to_expire: List[int],
    sigma: float,
    r: float = 0.05,
) -> pd.DataFrame:
    n_days = len(prices)
    results = []
    for strike in strikes:
        for days in days_to_expire:
            call_prices = []
            for i in range(n_days):
                S = prices[i]
                T = (days - i) / 252  # Update time to expiration
                if T >= 0:
                    call_price = black_scholes_call(S, strike, T, r, sigma)
                    call_prices.append(call_price)
                else:
                    call_prices.append(np.nan)  # Or any other appropriate value
            call_prices = np.array(call_prices)
            daily_pct_changes = np.diff(call_prices) / call_prices[:-1]
            daily_pct_changes = np.insert(
                daily_pct_changes, 0, np.nan
            )  # Insert 'NA' for day 0
            results.extend(
                zip(
                    np.repeat(days, n_days),
                    np.repeat(strike, n_days),
                    range(n_days),
                    call_prices,
                    daily_pct_changes,
                )
            )
    df = pd.DataFrame(
        results,
        columns=["Expire", "Strike", "Day", "Call Price", "Call Pct Change"],
    )
    return df


def gbm_prices(
    s0: float, mu: float, sigma: float, dt: float, n_steps: int
) -> np.ndarray:
    z = np.random.normal(size=n_steps)
    prices = s0 * np.exp(
        np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    )
    return prices


def generate_and_save_stock_data(s0, mu, sigma, dt, start_date, end_date, output_path):
    n_steps = (end_date - start_date).days + 1
    prices = gbm_prices(s0, mu, sigma, dt, n_steps)

    # Generate dates
    dates = [start_date + timedelta(days=int(i)) for i in range(n_steps)]

    # Save stock prices to CSV
    stock_data = pd.DataFrame({"date": dates, "price": prices})
    stock_data.to_csv(f"{output_path}/XYZ_stock.csv", index=False)


def generate_and_save_option_data(
    prices, start_date, r, sigma, strikes, expiry_dates, output_path
):
    n_steps = len(prices)
    df = call_prices(
        prices,
        strikes,
        [int((expiry_date - start_date).days) for expiry_date in expiry_dates],
        sigma,
        r,
    )

    # Generate dates
    dates = [start_date + timedelta(days=int(i)) for i in range(n_steps)]

    # Save option data to separate CSVs
    for strike in strikes:
        for expiry_date in expiry_dates:
            expiry_date_str = expiry_date.strftime("%y%m%d")
            strike_str = str(int(strike * 1000)).zfill(8)
            option_name = f"XYZ{expiry_date_str}C{strike_str}"
            days = (expiry_date - start_date).days
            option_data = df[(df["Strike"] == strike) & (df["Expire"] == days)]
            option_data = option_data[["Day", "Call Price", "Call Pct Change"]]
            option_data["Date"] = [dates[int(i)] for i in option_data["Day"]]
            option_data = option_data[["Date", "Call Price", "Call Pct Change"]]
            option_data.to_csv(f"{output_path}/{option_name}.csv", index=False)


def plot_asset_price(file_path):
    data = pd.read_csv(file_path)
    data["Date"] = pd.to_datetime(data["Date"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=data["Date"], y=data["Price"], mode="lines", name="Price")
    )
    fig.update_layout(
        title="Asset Price Over Time", xaxis_title="Date", yaxis_title="Price"
    )
    fig.show()


# ------------------------------------------------------------
# Black Scholes Model
# ------------------------------------------------------------


def black_scholes(S, K, T, r, sigma, option_type: Literal["call", "put"]):
    """
    Calculate the Black-Scholes option price for a single set of input parameters.

    Parameters:
    - S (float): Current stock price.
    - K (float): Strike price of the option.
    - T (float): Time to expiration (in years).
    - r (float): Risk-free interest rate (annualized).
    - sigma (float): Volatility of the underlying stock (annualized).
    - option_type (str): Type of the option - "call" or "put".

    Returns:
    - float: Calculated option price.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option_type. Must be 'call' or 'put'.")

    return price


def black_scholes_vectorized(
    S: Union[np.ndarray, pd.Series],
    K: float,
    T: Union[np.ndarray, pd.Series],
    r: float,
    sigma: float,
    option_type: str,
) -> Union[np.ndarray, pd.Series]:
    """
    Vectorized Black-Scholes formula.

    Parameters:
    - S: Stock price (np.ndarray or pd.Series)
    - K: Strike price (float)
    - T: Time to maturity in years (np.ndarray or pd.Series)
    - r: Risk-free interest rate (float)
    - sigma: Volatility of the underlying asset (float)
    - option_type: 'call' or 'put' (str)

    Returns:
    - Option price (np.ndarray or pd.Series)
    """
    # T = np.maximum(T, 1e-30)  # Avoid division by zero and ensure non-negative time
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option_type. Must be 'call' or 'put'.")

    return option_price


def calculate_option_prices(
    df: pd.DataFrame,
    expiration_date: str,
    strike_price: float,
    option_type: str,
    r: float,
    sigma: float,
) -> pd.DataFrame:
    """
    Calculate option prices for a DataFrame of stock prices using Black-Scholes model.

    Parameters:
    - df: DataFrame with columns ['date', 'price']
    - expiration_date: Option expiration date as string (YYYY-MM-DD)
    - strike_price: Strike price of the option (float)
    - option_type: Type of the option ('call' or 'put')
    - r: Risk-free interest rate (float)
    - sigma: Volatility of the underlying asset (float)

    Returns:
    - DataFrame with columns ['date', 'option_price']
    """
    df["date"] = pd.to_datetime(df["date"])
    expiration_date = pd.to_datetime(expiration_date)
    df["time_to_maturity"] = (expiration_date - df["date"]).dt.days / 365.0
    option_price = black_scholes_vectorized(
        df["price"].values,
        strike_price,
        df["time_to_maturity"].values,
        r,
        sigma,
        option_type,
    )
    option_df = pd.DataFrame({"date": df["date"], "price": option_price})

    # Fill value for expired options
    # df.loc[df["time_to_maturity"] <= 0, "option_price"] = np.nan

    return option_df


# ------------------------------------------------------------
# Option
# ------------------------------------------------------------


@dataclass
class Option:
    id: str
    option_type: Literal["call", "put"]
    strike: float
    expiry_date: datetime
    entry_date: datetime | None = None
    exit_date: datetime | None = None


def generate_options(
    strikes: np.ndarray,
    option_type: Literal["call", "put"],
    expiry_dates: list[datetime],
    underlying_stock_ticker: str = "XYZ",
) -> list[Option]:
    """
    Option id follows this: https://polygon.io/blog/how-to-read-a-stock-options-ticker
    """
    if option_type == "call":
        option_type_str = "C"
    elif option_type == "put":
        option_type_str = "P"
    else:
        raise Exception()

    options = []
    for strike in strikes:
        for expiry_date in expiry_dates:
            expiry_date_str = expiry_date.strftime("%y%m%d")
            strike_price_str = f"{int(abs(strike) * 1000):08d}"
            option_id = f"{underlying_stock_ticker}{expiry_date_str}{option_type_str}{strike_price_str}"
            option = Option(
                id=option_id,
                option_type=option_type,
                strike=strike,
                expiry_date=expiry_date,
            )
            options.append(option)
    return options


# -------- Rollover --------


def should_rollover(option: Option, current_date: datetime) -> bool:
    """Determine if the held option should be rolled over based on the current date."""
    return (option.expiry_date - current_date).days <= 10


def find_next_option(
    available_options: list[Option], current_date: datetime, current_market_price: float
) -> Option:
    """Find the next option to rollover to based on the criteria:
    1. Has at least 30 days before expiry.
    2. Is nearest to the current day.
    3. Is ATM (At-The-Money).
    """
    # Filter options that have at least 30 days before expiry
    valid_options = [
        option
        for option in available_options
        if (option.expiry_date - current_date).days >= 30
    ]
    if not valid_options:
        return None

    # Finding the option that is nearest to being At-The-Money (ATM)
    atm_option = min(
        valid_options,
        key=lambda x: (
            abs(x.strike - current_market_price),
            (x.expiry_date - current_date).days,
        ),
    )
    return atm_option


def rollover_options(
    available_options: list[Option], prices_df: pd.DataFrame
) -> list[Option]:
    prices_df["date"] = pd.to_datetime(prices_df["date"])

    rolled_over_options = []
    current_option = None

    for i, row in prices_df.iterrows():
        # current_date = datetime.strptime(row["date"], "%Y-%m-%d")
        current_date = row["date"]
        current_market_price = row["price"]

        if current_option is None or should_rollover(current_option, current_date):
            next_option = find_next_option(
                available_options, current_date, current_market_price
            )
            if next_option and (
                current_option is None or next_option.id != current_option.id
            ):
                if current_option:
                    current_option.exit_date = current_date
                next_option.entry_date = current_date

                rolled_over_options.append(next_option)
                print(
                    f"{current_date.strftime('%y-%m-%d')} {current_market_price} => Rolled over to option {next_option.strike}@{next_option.expiry_date.strftime('%y%m%d')}"
                )
                current_option = next_option  # Update the held option
            elif not next_option:
                print("No suitable option found to rollover to.")
                break  # Exit the loop if no suitable next option is found

    return rolled_over_options


# ------------------------------------------------------------
# Vectorized Backtest
# ------------------------------------------------------------


def prepare_market_data(
    options: List[Option],
    stock_prices: pd.DataFrame,
    r: float,
    sigma: float,
) -> Dict[str, pd.DataFrame]:
    """
    Prepare market data for options pricing.

    Parameters:
    - options: List of Option objects representing options
    - stock_prices: DataFrame with columns ['date', 'price'] representing underlying stock prices
    - r: Risk-free interest rate (float)
    - sigma: Volatility of the underlying asset (float)

    Returns:
    - Dictionary where the key is the asset id and the value is a DataFrame of the asset's prices
    """
    market_data = {}

    for option in options:
        option_prices = calculate_option_prices(
            stock_prices,
            option.expiry_date.strftime("%Y-%m-%d"),
            option.strike,
            option.option_type,
            r,
            sigma,
        )
        # option_prices["asset_id"] = option.id
        market_data[option.id] = option_prices

    return market_data


def rollovers_to_signals(rollovers, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    option_ids = list(OrderedDict.fromkeys(rollover.id for rollover in rollovers))
    signals_df = pd.DataFrame(0, index=date_range, columns=option_ids)

    # for option_id, entry_date, exit_date in rollovers:
    #     signals_df.loc[entry_date, option_id] = 1  # Buy signal
    #     signals_df.loc[exit_date, option_id] = -1  # Sell signal

    for option in rollovers:
        signals_df.loc[option.entry_date, option.id] = 1  # Buy signal
        signals_df.loc[option.exit_date, option.id] = -1  # Sell signal

    return signals_df


def generate_positions(signals_df):
    """
    Generate positions DataFrame from signals DataFrame using cumulative sum.
    Positions take effect one day after the signal, considering the cumulative effect.
    """
    # Use cumulative sum to accumulate signals
    positions_df = signals_df.cumsum()

    # Shift the positions to reflect that they take effect one day after the signal
    positions_df = positions_df.shift(1).fillna(0).astype(int)

    return positions_df


def calculate_returns(positions_df, market_data):
    """
    Calculate returns DataFrame from positions DataFrame and market data.
    """
    returns_df = pd.DataFrame(index=positions_df.index, columns=positions_df.columns)

    for asset_id in positions_df.columns:
        # asset_data = market_data[market_data["asset_id"] == asset_id].copy()
        asset_data = market_data[asset_id].copy()
        asset_data["date"] = pd.to_datetime(asset_data["date"])
        asset_data = asset_data.set_index("date")

        asset_positions = positions_df[asset_id]
        asset_returns = asset_positions * asset_data["price"].pct_change()
        returns_df[asset_id] = asset_returns

    return returns_df


def backtest(market_data, rollovers, start_date, end_date):
    """
    Perform backtest using market data and rollovers.
    """

    signals_df = rollovers_to_signals(rollovers, start_date, end_date)
    positions_df = generate_positions(signals_df)
    returns_df = calculate_returns(positions_df, market_data)

    return signals_df, positions_df, returns_df


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------

if __name__ == "__main__":
    output_path = "generated/XYZ"

    # Generate stock prices
    s0 = 100.0  # 初始價格
    mu = 0.05  # 預期收益率（年化）
    sigma = 0.2  # 波動率（年化）
    # dt = 1/252  # 時間步長（假設一年有252個交易日）
    # n_steps = 252 # 模擬步數（假設模擬一年的價格路徑）
    dt = 1 / 365  # 時間步長（假設一年有252個交易日）
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)

    generate_prices = True
    if generate_prices:
        generate_and_save_stock_data(
            s0, mu, sigma, dt, start_date, end_date, output_path
        )

    stock_prices_df = pd.read_csv(f"{output_path}/XYZ_stock.csv")
    prices = stock_prices_df["price"]

    # Generate options
    strikes = np.arange((min(prices) // 5 - 2) * 5, (max(prices) // 5 + 3) * 5 + 1, 5)
    expiry_dates = [start_date + timedelta(days=days) for days in range(30, 361, 30)]
    available_options = generate_options(strikes, "call", expiry_dates, "XYZ")

    # Backtest option strategy
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility

    rollovers = rollover_options(available_options, stock_prices_df)
    market_data = prepare_market_data(
        rollovers,
        stock_prices_df,
        r,
        sigma,
    )
    market_data["XYZ"] = stock_prices_df
    market_data["XYZ230131C00100000"]

    start_date = market_data["XYZ"]["date"].min()
    end_date = market_data["XYZ"]["date"].max()

    rollovers_to_signals(rollovers, start_date, end_date)

    # Perform backtest
    start_date = market_data["XYZ"]["date"].min()
    end_date = market_data["XYZ"]["date"].max()
    signals_df, positions_df, returns_df = backtest(
        market_data, rollovers, start_date, end_date
    )
    signals_df.to_csv(f"{output_path}/bt_signals.csv")
    positions_df.to_csv(f"{output_path}/bt_positions.csv")
    returns_df.to_csv(f"{output_path}/bt_returns.csv")

    # Analyze results
    # pd.options.display.max_rows = None  # 設定顯示的最大列數
    # pd.options.display.max_columns = None  # 設定顯示的最大欄位數
    # print("Signals:")
    # print(signals_df)
    #
    # print("\nPositions:")
    # print(positions_df)
    # print("\nReturns:")
    # print(returns_df)
