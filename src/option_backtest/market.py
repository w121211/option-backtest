import copy
import logging
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class Stock:
    id: str
    df: pd.DataFrame  # Columns = ['price']. Indexed by 'date'
    position: int = 0
    # entry_date: datetime | None = None
    # exit_date: datetime | None = None

    def __repr__(self):
        return f"Stock(id='{self.id}', position={self.position})"


@dataclass
class Option:
    id: str
    option_type: Literal["call", "put"]
    strike: float
    expiry_date: pd.Timestamp
    df: pd.DataFrame  # Columns ['price', 'delta', 'vega', ...], index by 'date'

    position: int = 0
    # entry_date: datetime | None = None
    # exit_date: datetime | None = None

    def __repr__(self):
        return f"Option(id='{self.id}', position={self.position})"


@dataclass
class Account:
    cash: float
    stock: Stock  # Underlying stock
    options: dict[str, Option]  # key: option id, value: option


@dataclass
class Trade:
    asset_type: Literal["stock", "option"]
    asset_id: str
    amount: int
    date: pd.Timestamp

@dataclass
class MarketState:
    stock: Stock  # Underlying stock
    listed_options: dict[str, Option]
    account: Account
    current_date: pd.Timestamp
    current_step: int


class OnTradeFn(Protocol):
    def __call__(self, state: MarketState) -> list[Trade]: ...


def execute_trades(
    state: MarketState,
    trades: list[Trade],
) -> Account:
    listed_options = state.listed_options
    account = state.account
    current_date = state.current_date

    for trade in trades:
        if trade.asset_type == "stock":
            stock = account.stock
            stock_price = stock.df.at[current_date, "price"]
            stock.position += trade.amount
            account.cash -= stock_price * trade.amount

            logger.info(
                f"Executed trade [{current_date.strftime("%Y-%m-%d")}] {trade.asset_id} @ {stock_price:.2f} * {trade.amount}, Stock position: {stock.position}, Cash: {account.cash:.2f}"
            )

        elif trade.asset_type == "option":
            if trade.asset_id not in account.options:
                opt = copy.copy(listed_options[trade.asset_id])
                opt.position = 0
                account.options[opt.id] = opt

            opt = account.options[trade.asset_id]
            opt_price = opt.df.at[current_date, "price"]
            opt.position += trade.amount
            account.cash -= opt_price * trade.amount * 100

            logger.info(
                f"Executed trade[{current_date.strftime("%Y-%m-%d")}] {trade.asset_id} @ {opt_price:.2f} * {trade.amount}, Option position: {opt.position}, Cash: {account.cash:.2f}"
            )

    return account


def after_market(
    state: MarketState
) -> tuple[dict[str, Option], Account]:
    listed_options = state.listed_options
    current_date = state.current_date
    account = state.account

    # Remove expired options from available options
    listed_options = {
        opt_id: opt
        for opt_id, opt in listed_options.items()
        if opt.expiry_date > current_date
    }

    # Check account options
    for opt_id, opt in list(account.options.items()):
        if opt.expiry_date <= current_date:
            current_stock_price = account.stock.df.at[current_date, "price"]

            # Option has expired
            if opt.option_type == "call":
                if current_stock_price > opt.strike and opt.position != 0:
                    # Exercise call option
                    account.stock.position += opt.position * 100
                    account.cash -= opt.position * opt.strike * 100
                    logger.info(
                        f"Exercised call option {opt_id}. Bought {opt.position * 100} shares at strike price {opt.strike}. Current stock price={current_stock_price}"
                    )

            elif opt.option_type == "put":
                if current_stock_price < opt.strike and opt.position != 0:
                    # Exercise put option
                    account.stock.position -= opt.position * 100
                    account.cash += opt.position * opt.strike * 100
                    logger.info(
                        f"Exercised put option {opt_id}. Sold {opt.position * 100} shares at strike price {opt.strike}."
                    )

            # Remove expired option from account
            del account.options[opt_id]
            # logger.info(f"Removed expired option {opt_id} from account.")

    return listed_options, account


def calculate_portfolio_value(account: Account, current_date: pd.Timestamp) -> float:
    portfolio_value = account.cash

    # Calculate stock value
    stock_price = account.stock.df.at[current_date, "price"]
    stock_value = account.stock.position * stock_price
    portfolio_value += stock_value

    # Calculate option values
    for opt in account.options.values():
        if opt.expiry_date > current_date:
            opt_price = opt.df.at[current_date, "price"]
            opt_value = opt.position * opt_price * 100
            portfolio_value += opt_value

    return portfolio_value


def episode(
    init_cash: float,
    listed_stock: Stock,
    listed_options: dict[str, Option],
    on_trade_fns: list[OnTradeFn],  # A list of trading functions
):
    account = Account(cash=init_cash, stock=listed_stock, options={})

    log_trades = []
    log_portfolio_values = []

    for i, date in enumerate(account.stock.df.index):
        state = MarketState(
            stock=listed_stock,
            listed_options=listed_options,
            account=account,
            current_date=date,
            current_step=i,
        )

        trades = []
        for on_trade in on_trade_fns:
            _trades = on_trade(state)
            trades.extend(_trades)

        account = execute_trades(state, trades)
        listed_options, account = after_market(state)
        portfolio_value = calculate_portfolio_value(account, date)

        log_trades.extend(trades)
        log_portfolio_values.append(portfolio_value)

    portfolio_df = pd.DataFrame(
        {"value": log_portfolio_values}, index=account.stock.df.index
    )

    return log_trades, portfolio_df
