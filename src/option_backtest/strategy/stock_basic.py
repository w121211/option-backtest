from typing import Callable, Literal, TypedDict

import pandas as pd

from ..market import Account, Option, Stock, Trade, TradeActionFn

# --- Conditions ---


def if_stock_price_above(
    price: float,
) -> Callable[[Stock, Account, pd.Timestamp], bool]:
    def condition(stock: Stock, account: Account, current_date: pd.Timestamp) -> bool:
        current_price = stock.df.at[current_date, "price"]
        return current_price > price

    return condition


def if_stock_price_below(
    price: float,
) -> Callable[[Stock, Account, pd.Timestamp], bool]:
    def condition(stock: Stock, account: Account, current_date: pd.Timestamp) -> bool:
        current_price = stock.df.at[current_date, "price"]
        return current_price < price

    return condition


# --- Actions ---


def buy_stock(
    position: int,
) -> Callable[[Stock, Account, pd.Timestamp], list[Trade]]:
    def action(
        stock: Stock, account: Account, current_date: pd.Timestamp
    ) -> list[Trade]:
        return [
            Trade(
                asset_type="stock",
                asset_id=stock.id,
                position=position,
                date=current_date,
            ),
        ]

    return action


def sell_stock(
    position: int,
) -> Callable[[Stock, Account, pd.Timestamp], list[Trade]]:
    def action(
        stock: Stock, account: Account, current_date: pd.Timestamp
    ) -> list[Trade]:
        return [
            Trade(
                asset_type="stock",
                asset_id=stock.id,
                position=-position,
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
                position=-account.stock.position,
                date=current_date,
            ),
        ]

    return action


# --- Strategy ---


def create_stock_strategy(
    condition_action_pairs: list[
        tuple[
            Callable[[Stock, Account, pd.Timestamp], bool],
            Callable[[Stock, Account, pd.Timestamp], list[Trade]],
        ]
    ],
    default_action: Callable[[Stock, Account, pd.Timestamp], list[Trade]] | None = None,
) -> TradeActionFn:
    def strategy(
        listed_options: dict[str, Option],
        account: Account,
        current_date: pd.Timestamp,
    ) -> list[Trade]:
        trades: list[Trade] = []
        stock = account.stock

        for condition, action in condition_action_pairs:
            if condition(stock, account, current_date):
                trades.extend(action(stock, account, current_date))
                break

        if not trades and default_action is not None:
            trades.extend(default_action(stock, account, current_date))

        return trades

    return strategy


# --- Example usage ---

if __name__ == "__main__":
    condition_action_pairs = [
        (if_stock_price_above(100), buy_stock(100)),
        (if_stock_price_below(90), close_stock_position()),
    ]

    stock_strategy = create_stock_strategy(condition_action_pairs)
