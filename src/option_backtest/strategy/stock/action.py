from typing import Callable

import pandas as pd

from ...market import Account, Option, Stock, Trade, TradeActionFn


def buy_stock(
    amount: int,
) -> Callable[[Stock, Account, pd.Timestamp], list[Trade]]:
    """Negative amount represent sell."""

    def action(
        stock: Stock, account: Account, current_date: pd.Timestamp
    ) -> list[Trade]:
        return [
            Trade(
                asset_type="stock",
                asset_id=stock.id,
                amount=amount,
                date=current_date,
            ),
        ]

    return action


def close_stock_position() -> Callable[[Stock, Account, pd.Timestamp], list[Trade]]:
    def action(
        stock: Stock, account: Account, current_date: pd.Timestamp
    ) -> list[Trade]:
        return [
            Trade(
                asset_type="stock",
                asset_id=stock.id,
                amount=-account.stock.position,
                date=current_date,
            ),
        ]

    return action
