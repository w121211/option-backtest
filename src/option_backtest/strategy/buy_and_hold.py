import pandas as pd

from ..bt_event import BacktestConfig, run_backtest
from ..market import Account, OnTradeFn, Option, MarketState, Trade


def buy_stock_on_day_0(amount: int) -> OnTradeFn:
    def _buy_stock_on_day_0(state: MarketState) -> list[Trade]:
        if state.current_step == 0:
            return [Trade("stock", state.account.stock.id, amount, state.current_date)]
        return []

    return _buy_stock_on_day_0
