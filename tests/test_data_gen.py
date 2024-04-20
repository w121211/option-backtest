from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.option_backtest.market import Stock
from src.option_backtest.gen_data import (
    black_scholes_vectorized,
    generate_options,
)


# -------- Test Black Scholes ------


def test_black_scholes_vectorized_call():
    S = np.array([100, 100, 100])
    K = 100
    T = np.array([1, 0.5, 0])
    r = 0.05
    sigma = 0.2
    option_type = "call"

    expected_price = np.array([10.4506, 6.8887, 0])
    expected_delta = np.array([0.6368, 0.5977, 0])

    price, delta = black_scholes_vectorized(S, K, T, r, sigma, option_type)

    np.testing.assert_almost_equal(price, expected_price, decimal=4)
    np.testing.assert_almost_equal(delta, expected_delta, decimal=4)


def test_black_scholes_vectorized_put():
    S = np.array([100, 100, 100])
    K = 100
    T = np.array([1, 0.5, 0])
    r = 0.05
    sigma = 0.2
    option_type = "put"

    expected_price = np.array([5.5735, 4.4197, 0])
    expected_delta = np.array([-0.3632, -0.4023, 0])

    price, delta = black_scholes_vectorized(S, K, T, r, sigma, option_type)

    np.testing.assert_almost_equal(price, expected_price, decimal=4)
    np.testing.assert_almost_equal(delta, expected_delta, decimal=4)


def test_black_scholes_vectorized_pandas():
    S = pd.Series([100, 100, 100])
    K = 100
    T = pd.Series([1, 0.5, 0])
    r = 0.05
    sigma = 0.2
    option_type = "call"

    expected_price = pd.Series([10.4506, 6.8887, 0])
    expected_delta = pd.Series([0.6368, 0.5977, 0])

    price, delta = black_scholes_vectorized(S, K, T, r, sigma, option_type)

    pd.testing.assert_series_equal(price, expected_price, atol=1e-4)
    pd.testing.assert_series_equal(delta, expected_delta, atol=1e-4)


def test_black_scholes_vectorized_invalid_option_type():
    S = np.array([100])
    K = 100
    T = np.array([1])
    r = 0.05
    sigma = 0.2
    option_type = "invalid"

    with pytest.raises(
        ValueError, match="Invalid option_type. Must be 'call' or 'put'."
    ):
        black_scholes_vectorized(S, K, T, r, sigma, option_type)


def test_black_scholes_vectorized_with_nonpositive_time_to_maturity():
    S = np.array([100, 100, 100])
    K = 100
    T = np.array([1, 0, -0.5])  # Mix of positive, zero, and negative time to maturity
    r = 0.05
    sigma = 0.2
    option_type = "call"

    expected_price = np.array([10.4506, 0, 0])
    expected_delta = np.array([0.6368, 0, 0])

    price, delta = black_scholes_vectorized(S, K, T, r, sigma, option_type)

    np.testing.assert_almost_equal(price, expected_price, decimal=4)
    np.testing.assert_almost_equal(delta, expected_delta, decimal=4)


@pytest.mark.parametrize(
    "T_values, expected_results",
    [
        (
            # T values
            np.array([-1, 0, 0.5, 1, 2]),
            # Expected option prices
            np.array([0.0, 0.0, 6.88872858, 10.45058357, 16.12677972]),
        ),
        (
            # T values reorder
            np.array([0.5, -1, 1, 0, 2]),
            # Expected option prices
            np.array([6.88872858, 0.0, 10.45058357, 0.0, 16.12677972]),
        ),
    ],
)
def test_black_scholes_time_ranges(T_values, expected_results):
    S = np.array([100, 100, 100, 100, 100])  # Constant stock price for simplicity
    K = 100  # Strike price
    r = 0.05  # Risk-free interest rate
    sigma = 0.2  # Volatility
    option_type = "call"  # Testing with call option

    # Call the vectorized function
    price, delta = black_scholes_vectorized(S, K, T_values, r, sigma, option_type)

    # Compare the expected and calculated prices
    np.testing.assert_allclose(
        price,
        expected_results,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Vectorized results should match expected true values",
    )


# -------- Test Generate Options ------

# def test_generate_options(snapshot):
#     # Define test input parameters
#     strikes = np.array([50, 60, 70])
#     expiry_dates = [
#         datetime(2023, 6, 30),
#         datetime(2023, 9, 30),
#     ]
#     underlying_stock_ticker = "XYZ"

#     # Call the function to generate options
#     options = generate_options(strikes, expiry_dates, underlying_stock_ticker)

#     assert options == snapshot


def test_generate_options(snapshot):
    # Create a sample stock object
    stock_df = pd.DataFrame(
        {"price": [100, 110, 120]},
        index=pd.date_range(start="2023-01-01", periods=3, freq="d"),
    )
    stock = Stock(id="XYZ", df=stock_df)

    # Define the input parameters
    strikes = np.array([90, 100, 110])
    expiry_dates = [pd.Timestamp("2023-06-30"), pd.Timestamp("2023-12-31")]
    r = 0.05
    sigma = 0.2

    # Call the generate_options function
    options = generate_options(stock, strikes, expiry_dates, r, sigma)
    opt_0_df = list(options.values())[0].df

    assert options == snapshot
    assert opt_0_df.to_string() == snapshot
