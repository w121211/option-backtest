from datetime import timedelta
from typing import Callable

import pandas as pd

from ...market import Account, Option, Stock, Trade, TradeActionFn


def calculate_sma(period: int) -> Callable[[Stock, Account, pd.Timestamp], float]:
    def sma(stock: Stock, account: Account, current_date: pd.Timestamp) -> float:
        price_history = stock.df.loc[:current_date, "price"]
        return price_history[-period:].mean()

    return sma


def if_price_above_sma(
    period: int,
) -> Callable[[Stock, Account, pd.Timestamp], bool]:
    def condition(stock: Stock, account: Account, current_date: pd.Timestamp) -> bool:
        current_price = stock.df.at[current_date, "price"]
        sma = calculate_sma(period)(stock, account, current_date)
        return current_price > sma

    return condition


def if_price_below_sma(
    period: int,
) -> Callable[[Stock, Account, pd.Timestamp], bool]:
    def condition(stock: Stock, account: Account, current_date: pd.Timestamp) -> bool:
        current_price = stock.df.at[current_date, "price"]
        sma = calculate_sma(period)(stock, account, current_date)
        return current_price < sma

    return condition


def if_price_crosses_above_sma(
    period: int,
) -> Callable[[Stock, Account, pd.Timestamp], bool]:
    def condition(stock: Stock, account: Account, current_date: pd.Timestamp) -> bool:
        current_price = stock.df.at[current_date, "price"]
        prev_price = stock.df.at[current_date - pd.Timedelta(days=1), "price"]
        sma = calculate_sma(period)(stock, account, current_date)
        prev_sma = calculate_sma(period)(
            stock, account, current_date - pd.Timedelta(days=1)
        )
        return prev_price <= prev_sma and current_price > sma

    return condition


def if_price_crosses_below_sma(
    period: int,
) -> Callable[[Stock, Account, pd.Timestamp], bool]:
    def condition(stock: Stock, account: Account, current_date: pd.Timestamp) -> bool:
        current_price = stock.df.at[current_date, "price"]
        prev_price = stock.df.at[current_date - pd.Timedelta(days=1), "price"]
        sma = calculate_sma(period)(stock, account, current_date)
        prev_sma = calculate_sma(period)(
            stock, account, current_date - pd.Timedelta(days=1)
        )
        return prev_price >= prev_sma and current_price < sma

    return condition


def if_price_above_ema(
    period: int,
) -> Callable[[Stock, Account, pd.Timestamp], bool]:
    def condition(stock: Stock, account: Account, current_date: pd.Timestamp) -> bool:
        current_price = stock.df.at[current_date, "price"]
        ema = stock.df.loc[:current_date, "price"].ewm(span=period).mean().iloc[-1]
        return current_price > ema

    return condition


def if_price_below_ema(
    period: int,
) -> Callable[[Stock, Account, pd.Timestamp], bool]:
    def condition(stock: Stock, account: Account, current_date: pd.Timestamp) -> bool:
        current_price = stock.df.at[current_date, "price"]
        ema = stock.df.loc[:current_date, "price"].ewm(span=period).mean().iloc[-1]
        return current_price < ema

    return condition


def if_macd_crosses_above_signal(
    fast_period: int,
    slow_period: int,
    signal_period: int,
) -> Callable[[Stock, Account, pd.Timestamp], bool]:
    def condition(stock: Stock, account: Account, current_date: pd.Timestamp) -> bool:
        price_history = stock.df.loc[:current_date, "price"]
        macd = (
            price_history.ewm(span=fast_period).mean()
            - price_history.ewm(span=slow_period).mean()
        )
        signal = macd.ewm(span=signal_period).mean()
        return macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]

    return condition


def if_macd_crosses_below_signal(
    fast_period: int,
    slow_period: int,
    signal_period: int,
) -> Callable[[Stock, Account, pd.Timestamp], bool]:
    def condition(stock: Stock, account: Account, current_date: pd.Timestamp) -> bool:
        price_history = stock.df.loc[:current_date, "price"]
        macd = (
            price_history.ewm(span=fast_period).mean()
            - price_history.ewm(span=slow_period).mean()
        )
        signal = macd.ewm(span=signal_period).mean()
        return macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]

    return condition
