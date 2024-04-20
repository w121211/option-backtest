from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm


# -------- Generate Stock Prices by Gometrical Brownian Motion --------


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
    stock_data.to_csv(output_path, index=False)


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


# -------- Black Scholes Model --------


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
    Vectorized Black-Scholes formula, modified to handle cases where T <= 0
    by setting the option price to zero.

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

    # Initialize option_price as zeros
    if isinstance(T, np.ndarray):
        option_price = np.zeros_like(T, dtype=np.float64)
    elif isinstance(T, pd.Series):
        option_price = pd.Series(np.zeros(len(T)), dtype=np.float64, index=T.index)

    # Ensure T is non-negative, else set option price to 0
    mask_T = T > 0
    S_filtered = S[mask_T]  # Filter S as well to match the filtered T's dimensions
    T_filtered = T[mask_T]

    if np.any(mask_T):  # Proceed only if there are any valid T values
        d1 = (np.log(S_filtered / K) + (r + 0.5 * sigma**2) * T_filtered) / (
            sigma * np.sqrt(T_filtered)
        )
        d2 = d1 - sigma * np.sqrt(T_filtered)

        if option_type == "call":
            option_price[mask_T] = S_filtered * norm.cdf(d1) - K * np.exp(
                -r * T_filtered
            ) * norm.cdf(d2)
        elif option_type == "put":
            option_price[mask_T] = K * np.exp(-r * T_filtered) * norm.cdf(
                -d2
            ) - S_filtered * norm.cdf(-d1)
        else:
            raise ValueError("Invalid option_type. Must be 'call' or 'put'.")
    # For T <= 0, option_price remains zero as initialized

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
    position: int | None = None
    entry_date: datetime | None = None
    exit_date: datetime | None = None

    def __repr__(self):
        entry_date_str = (
            self.entry_date.strftime("%Y-%m-%d") if self.entry_date else None
        )
        exit_date_str = self.exit_date.strftime("%Y-%m-%d") if self.exit_date else None
        expiry_date_str = self.expiry_date.strftime("%Y-%m-%d")

        return f"Option(id='{self.id}', option_type='{self.option_type}', strike={self.strike}, expiry_date='{expiry_date_str}', position={self.position}, entry_date='{entry_date_str}', exit_date='{exit_date_str}')"


def generate_options(
    strikes: np.ndarray,
    expiry_dates: list[datetime],
    underlying_stock_ticker: str = "XYZ",
) -> list[Option]:
    """
    Generate a list of call and put options with the given strikes and expiry dates.
    Option id follows this: https://polygon.io/blog/how-to-read-a-stock-options-ticker

    Parameters:
    - strikes: Array of strike prices for the options.
    - expiry_dates: List of expiry dates for the options.
    - underlying_stock_ticker: Ticker symbol of the underlying stock (default: "XYZ").

    Returns:
    - List of Option objects representing the generated call and put options.
    """
    options = []
    for expiry_date in expiry_dates:
        for option_type in ["call", "put"]:
            for strike in strikes:
                expiry_date_str = expiry_date.strftime("%y%m%d")
                strike_price_str = f"{int(abs(strike) * 1000):08d}"
                option_id = f"{underlying_stock_ticker}{expiry_date_str}{'C' if option_type == 'call' else 'P'}{strike_price_str}"
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


def rollover_short_atm_calls(
    available_options: list[Option], prices_df: pd.DataFrame
) -> list[Option]:
    """
    Implement a short ATM (At-The-Money) call option rollover strategy.

    The strategy sells the nearest ATM call option with at least 30 days to expiry
    and rolls over to the next ATM call option when there are 10 days or fewer left to expiry.

    Parameters:
    - available_options: List of available Option objects to choose from.
    - prices_df: DataFrame with columns ['date', 'price'] representing the underlying asset prices.

    Returns:
    - List of Option objects representing the rolled-over short call options.
    """
    prices_df["date"] = pd.to_datetime(prices_df["date"])

    rolled_over_options = []
    current_option = None

    for i, row in prices_df.iterrows():
        current_date = row["date"]
        current_market_price = row["price"]

        if (
            current_option is None
            or (current_option.expiry_date - current_date).days <= 10
        ):
            # Find the next ATM call option to sell
            valid_options = [
                option
                for option in available_options
                if (option.expiry_date - current_date).days >= 30
                and option.option_type == "call"
            ]
            if valid_options:
                next_option = min(
                    valid_options,
                    key=lambda x: (
                        abs(x.strike - current_market_price),
                        (x.expiry_date - current_date).days,
                    ),
                )
                if current_option is None or next_option.id != current_option.id:
                    if current_option:
                        current_option.exit_date = current_date
                        # current_option.position = 0  # Close the current short position
                    next_option.entry_date = current_date
                    next_option.position = -1  # Open a new short position

                    rolled_over_options.append(next_option)
                    print(
                        f"{current_date.strftime('%y-%m-%d')} {current_market_price} => Rolled over to selling option {next_option.strike}@{next_option.expiry_date.strftime('%y%m%d')}"
                    )
                    current_option = next_option  # Update the sold option
            else:
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


def rollovers_to_signals(
    rollovers: List[Option], start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    option_ids = list(OrderedDict.fromkeys(rollover.id for rollover in rollovers))
    signals_df = pd.DataFrame(0, index=date_range, columns=option_ids)

    for option in rollovers:
        if option.position is None:
            raise Exception(f"Option {option.id} has a None position.")

        signals_df.loc[option.entry_date, option.id] = (
            option.position
        )  # All options should have the entry
        if option.exit_date:
            signals_df.loc[option.exit_date, option.id] = -option.position

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
    Calculate daily percentage returns positions DataFrame and market data.
    """
    returns_df = pd.DataFrame(index=positions_df.index, columns=positions_df.columns)

    for asset_id in positions_df.columns:
        asset_df = market_data[asset_id].copy()
        asset_df["date"] = pd.to_datetime(asset_df["date"])
        asset_df = asset_df.set_index("date")

        asset_positions = positions_df[asset_id]
        asset_returns = asset_positions * asset_df["price"].pct_change(fill_method=None)
        returns_df[asset_id] = asset_returns

    return returns_df


def calculate_portfolio_values(
    positions_df: pd.DataFrame,
    market_data: Dict[str, pd.DataFrame],
    initial_cash: float = 0.0,
) -> pd.DataFrame:
    """
    Calculate portfolio values based on positions and market data.
    """
    # Merge market data and positions into a single DataFrame
    portfolio_df = None
    for asset_id, asset_df in market_data.items():
        asset_df["date"] = pd.to_datetime(asset_df["date"])
        asset_df = asset_df.set_index("date")
        asset_df = asset_df.rename(columns={"price": f"{asset_id}_price"})
        asset_df[f"{asset_id}_position"] = positions_df[asset_id]

        if portfolio_df is None:
            portfolio_df = asset_df
        else:
            portfolio_df = portfolio_df.join(asset_df, how="outer")

    # portfolio_df = portfolio_df.fillna(method="ffill")

    for asset_id in positions_df.columns:
        # Calculate daily changes in positions
        portfolio_df[f"{asset_id}_position_change"] = (
            portfolio_df[f"{asset_id}_position"].diff().fillna(0)
        )
        # Calculate cash flow for each asset
        portfolio_df[f"{asset_id}_cash_flow"] = (
            -portfolio_df[f"{asset_id}_position_change"]
            * portfolio_df[f"{asset_id}_price"]
        )
        # Calculate daily asset values
        portfolio_df[f"{asset_id}_value"] = (
            portfolio_df[f"{asset_id}_position"] * portfolio_df[f"{asset_id}_price"]
        )

    # Calculate cumulative cash position
    cash_flow_columns = [f"{asset_id}_cash_flow" for asset_id in positions_df.columns]
    portfolio_df["cash_balance"] = (
        portfolio_df[cash_flow_columns].sum(axis=1).cumsum() + initial_cash
    )

    # Calculate total portfolio value
    asset_value_columns = [f"{asset_id}_value" for asset_id in positions_df.columns]
    portfolio_df["portfolio_value"] = (
        portfolio_df[asset_value_columns].sum(axis=1) + portfolio_df["cash_balance"]
    )

    return portfolio_df


def plot_results(
    market_data: Dict[str, pd.DataFrame],
    portfolio_df: pd.DataFrame,
    rollovers: List[Option],
):
    """
    Plot the portfolio value, stock price, and rolled-over options.

    Parameters:
    - market_data: Dictionary of market data DataFrames, where the key is the asset ID and the value is a DataFrame with columns 'date' and 'price'.
    - portfolio_df: DataFrame of portfolio values and cash balances.
    - rollovers: List of Option objects representing the rolled-over options.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(portfolio_df["portfolio_value"], color="blue", label="Portfolio Value")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Portfolio Value", color="blue")
    ax1.tick_params("y", colors="blue")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(
        market_data["XYZ"].set_index("date")["price"], color="red", label="Stock Price"
    )
    ax2.set_ylabel("Stock Price", color="red")
    ax2.tick_params("y", colors="red")
    ax2.legend(loc="upper right")

    # Add markers for rolled-over options
    for option in rollovers:
        if option.entry_date:
            ax2.scatter(
                option.entry_date,
                market_data["XYZ"].set_index("date").loc[option.entry_date, "price"],
                marker="o",
                color="green",
                s=50,
                label=f"Entry: {option.strike}",
            )

    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.title("Portfolio Value, Stock Price, and Rolled-over Options")
    plt.show()


def backtest_dev():
    # Parameters
    ...

    # Read or generate stock prices

    # Generate available options

    #

    # Prepare market data: 1. stock prices
    pass

    #


def backtest(
    # market_data: Dict[str, pd.DataFrame],
    # rollovers: List[Option],
    # start_date: datetime,
    # end_date: datetime,
    output_dir: str,
    stock_ticker: str,
    stock_prices: pd.DataFrame,
    initial_cash: float = 0.0,
) -> Tuple[
    List[Option], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
]:
    """
    Perform backtest using market data and rollovers.

    Parameters:
    - market_data: Dictionary of market data DataFrames, where the key is the asset ID and the value is a DataFrame with columns 'date' and 'price'.
    - start_date: Start date of the backtest (datetime).
    - end_date: End date of the backtest (datetime).
    - initial_cash: Initial cash balance (default: 0).

    Returns:
    - rollovers: List of Option objects representing the rolled-over options.
    - signals_df: DataFrame of position signals.
    - positions_df: DataFrame of positions.
    - portfolio_df: DataFrame of portfolio values and cash balances.
    - portfolio_returns: Series of portfolio returns.
    - cumulative_returns: Series of cumulative portfolio returns.
    """
    output_dir = f"output/{stock_ticker}"

    # Read or generate stock prices
    s0 = 100.0  # 初始價格
    mu = 0.05  # 預期收益率（年化）
    sigma = 0.2  # 波動率（年化）
    dt = 1 / 252  # 時間步長（假設一年有252個交易日）
    n_steps = 252  # 模擬步數（假設模擬一年的價格路徑）
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)

    generate_prices = True
    if generate_prices:
        generate_and_save_stock_data(
            s0,
            mu,
            sigma,
            dt,
            start_date,
            end_date,
            f"{output_dir}/{stock_ticker}_prices.csv",
        )

    stock_prices_df = pd.read_csv(f"{output_dir}/{stock_ticker}_prices.csv")

    # Generate available options
    strikes = np.arange(
        (min(stock_prices_df["price"]) // 5 - 2) * 5,
        (max(stock_prices_df["price"]) // 5 + 3) * 5 + 1,
        5,
    )
    expiry_dates = [start_date + timedelta(days=days) for days in range(30, 361, 30)]
    available_options = generate_options(strikes, expiry_dates, "XYZ")

    # Backtest option strategy
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility

    # Roll over strategy
    # rollovers = rollover_options(available_options, stock_prices_df)
    rollovers = rollover_short_atm_calls(available_options, stock_prices_df)

    market_data = prepare_market_data(
        rollovers,
        stock_prices_df,
        r,
        sigma,
    )

    signals_df = rollovers_to_signals(rollovers, start_date, end_date)
    positions_df = generate_positions(signals_df)

    # Add underlying asset
    market_data[stock_ticker] = stock_prices_df
    positions_df[stock_ticker] = 0.0

    portfolio_df = calculate_portfolio_values(positions_df, market_data, initial_cash)

    # Calculate portfolio returns
    portfolio_returns = portfolio_df["portfolio_value"].pct_change()
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # Calculate performance metrics
    total_return = cumulative_returns.iloc[-1] - 1
    sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()

    # Print performance metrics
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")

    return (
        rollovers,
        signals_df,
        positions_df,
        portfolio_df,
        portfolio_returns,
        cumulative_returns,
    )


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------

if __name__ == "__main__":
    stock_ticker = "XYZ"
    output_dir = f"output/{stock_ticker}"

    # Perform backtest
    # start_date = market_data["XYZ"]["date"].min()
    # end_date = market_data["XYZ"]["date"].max()
    (
        rollovers,
        signals_df,
        positions_df,
        portfolio_df,
        portfolio_returns,
        cumulative_returns,
    ) = backtest(output_dir, initial_cash=100.0, stock_ticker=stock_ticker)

    # Analyze results
    # plot_results(portfolio_df, rollovers)

    # for k, v in market_data.items():
    #     v.to_csv(f"{output_path}/mk_{k}.csv")
    portfolio_df.to_csv(f"{output_dir}/bt_portfolio_df.csv")

    # signals_df.to_csv(f"{output_path}/bt_signals.csv")
    # positions_df.to_csv(f"{output_path}/bt_positions.csv")
    # daily_returns_df.to_csv(f"{output_path}/bt_daily_returns_df.csv")
    # portfolio_returns.to_csv(f"{output_path}/bt_portfolio_returns.csv")
    # cumulative_returns.to_csv(f"{output_path}/bt_cumulative_returns.csv")
