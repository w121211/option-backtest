from datetime import datetime, timedelta
from typing import Dict, List, Literal, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

from .market import Option, Stock


# -------- Generate Market Data --------


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


def generate_and_save_stock_data(
    s0: float,
    mu: float,
    sigma: float,
    dt: float,
    start_date: datetime,
    end_date: datetime,
    output_path: str,
):
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


def black_scholes_vectorized_old(
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


def black_scholes_vectorized(
    S: Union[np.ndarray, pd.Series],
    K: float,
    T: Union[np.ndarray, pd.Series],
    r: float,
    sigma: float,
    option_type: str,
) -> Tuple[Union[np.ndarray, pd.Series], Union[np.ndarray, pd.Series]]:
    """
    Vectorized Black-Scholes formula, modified to handle cases where T <= 0
    by setting the option price and delta to zero.

    Parameters:
    - S: Stock price (np.ndarray or pd.Series)
    - K: Strike price (float)
    - T: Time to maturity in years (np.ndarray or pd.Series)
    - r: Risk-free interest rate (float)
    - sigma: Volatility of the underlying asset (float)
    - option_type: 'call' or 'put' (str)

    Returns:
    - Option price (np.ndarray or pd.Series)
    - Option delta (np.ndarray or pd.Series)
    """

    # Initialize option_price and option_delta as zeros
    if isinstance(T, np.ndarray):
        option_price = np.zeros_like(T, dtype=np.float64)
        option_delta = np.zeros_like(T, dtype=np.float64)
    elif isinstance(T, pd.Series):
        option_price = pd.Series(np.zeros(len(T)), dtype=np.float64, index=T.index)
        option_delta = pd.Series(np.zeros(len(T)), dtype=np.float64, index=T.index)

    # Ensure T is non-negative, else set option price and delta to 0
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
            option_delta[mask_T] = norm.cdf(d1)
        elif option_type == "put":
            option_price[mask_T] = K * np.exp(-r * T_filtered) * norm.cdf(
                -d2
            ) - S_filtered * norm.cdf(-d1)
            option_delta[mask_T] = norm.cdf(d1) - 1
        else:
            raise ValueError("Invalid option_type. Must be 'call' or 'put'.")
    # For T <= 0, option_price and option_delta remain zero as initialized

    return option_price, option_delta


def calculate_option_price(
    stock_price_df: pd.DataFrame,
    expiry_date: pd.Timestamp,
    strike: float,
    option_type: str,
    r: float,
    sigma: float,
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """
    Calculate option prices for a given set of stock prices using the Black-Scholes model.

    Parameters:
    - stock_prices: DataFrame containing stock prices with columns ['price'] and indexed by 'date'
    - expiry_date: Expiration date of the option (datetime object)
    - strike: Strike price of the option
    - option_type: Type of the option ('call' or 'put')
    - r: Risk-free interest rate (annualized)
    - sigma: Volatility of the underlying asset (annualized)
    - trading_days_per_year: Number of trading days in a year (default: 252)

    Returns:
    - DataFrame containing option data with columns ['price', 'delta'] and indexed by 'date'
    """
    # Calculate time to maturity for each stock price date
    time_to_maturity = (
        expiry_date - stock_price_df.index
    ).to_series().dt.days / trading_days_per_year

    # Calculate option prices using the Black-Scholes model
    option_price, option_delta = black_scholes_vectorized(
        stock_price_df["price"].values,
        strike,
        time_to_maturity.values,
        r,
        sigma,
        option_type,
    )

    # Create a new DataFrame with date and option price columns
    option_df = pd.DataFrame(
        {
            # "date": stock_price_df["date"],
            "price": option_price,
            "delta": option_delta,
        },
        index=stock_price_df.index,
    )
    # option_price_df = option_price_df.set_index("date")
    return option_df


# ------------------------------------------------------------
# Option
# ------------------------------------------------------------


def generate_options(
    stock: Stock,
    strikes: np.ndarray,
    expiry_dates: list[pd.Timestamp],
    r: float,
    sigma: float,
) -> dict[str, Option]:
    """
    Generate a dictionary of call and put options for the given stock, with the specified strike prices and expiry dates.

    The option ID is constructed following the convention described in:
    https://polygon.io/blog/how-to-read-a-stock-options-ticker

    Parameters:
    - stock: The underlying Stock object for which the options are generated.
    - strikes: A NumPy array of strike prices for the options.
    - expiry_dates: A list of expiry dates for the options, represented as pd.Timestamp objects.
    - r: The risk-free interest rate used for option pricing.
    - sigma: The volatility of the underlying stock used for option pricing.

    Returns:
    - A dictionary of Option objects, where the keys are the option IDs and the values are the corresponding Option objects.
    """
    options: dict[str, Option] = {}
    for expiry_date in expiry_dates:
        for opt_type in ["call", "put"]:
            for strike in strikes:
                expiry_date_str = expiry_date.strftime("%y%m%d")
                strike_price_str = f"{int(abs(strike) * 1000):08d}"
                opt_id = f"{stock.id}{expiry_date_str}{'C' if opt_type == 'call' else 'P'}{strike_price_str}"
                opt_df = calculate_option_price(
                    stock.df, expiry_date, strike, opt_type, r, sigma
                )
                opt = Option(
                    id=opt_id,
                    option_type=opt_type,
                    strike=strike,
                    expiry_date=expiry_date,
                    df=opt_df,
                )
                options[opt.id] = opt
    return options
