import logging
from typing import Callable, Literal, TypedDict

import pandas as pd

from ..market import Account, MarketState, Option, Trade, OnTradeFn
from ..option import FindOptionsBy, find_options


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Helpers ---


class RolloverOptionNotFoundError(Exception):
    """Exception raised when the next rollover option is not found."""

    pass


def count_option_legs(account: Account) -> int:
    legs = 0
    for option in account.options.values():
        if option.position is not None and option.position != 0:
            legs += 1
    return legs


def get_current_option(account: Account) -> Option | None:
    for option in account.options.values():
        if option.position != 0:
            return option
    return None


# --- Composed rollover strategy ---

RolloverCondition = Callable[[Option, Account, pd.Timestamp], bool]
RolloverAction = Callable[[dict[str, Option], Account, pd.Timestamp], list[Trade]]

# --- Conditions ---

DeltaCondition = Callable[[float], bool]


def if_delta(condition: DeltaCondition) -> RolloverCondition:
    def rollover_condition(
        current_option: Option, account: Account, current_date: pd.Timestamp
    ) -> bool:
        current_delta = current_option.df.at[current_date, "delta"]
        return condition(current_delta)

    return rollover_condition


def if_expiration_within_days(days: int) -> RolloverCondition:
    def condition(
        current_option: Option, account: Account, current_date: pd.Timestamp
    ) -> bool:
        dte = (current_option.expiry_date - current_date).days
        return dte < days

    return condition


# --- Actions ---


def rollover_to(find_options_by: FindOptionsBy, buy_amount: int) -> RolloverAction:
    def action(
        listed_options: dict[str, Option], account: Account, current_date: pd.Timestamp
    ) -> list[Trade]:
        current_option = get_current_option(account)

        if current_option is None:
            raise Exception("Current option not found")

        current_stock_price = account.stock.df.at[current_date, "price"]

        found_options = find_options(
            listed_options,
            current_stock_price,
            current_date,
            find_options_by,
        )
        if not found_options:
            raise RolloverOptionNotFoundError("Not able to find next rollover option")
        else:
            next_option = found_options[0]
            return [
                Trade(
                    asset_type="option",
                    asset_id=current_option.id,
                    amount=-buy_amount,
                    date=current_date,
                ),
                Trade(
                    asset_type="option",
                    asset_id=next_option.id,
                    amount=buy_amount,
                    date=current_date,
                ),
            ]

    return action


def buy_option(find_options_by: FindOptionsBy, amount: int) -> RolloverAction:
    def action(
        listed_options: dict[str, Option], account: Account, current_date: pd.Timestamp
    ) -> list[Trade]:
        current_stock_price = account.stock.df.at[current_date, "price"]

        found_options = find_options(
            listed_options, current_stock_price, current_date, find_options_by
        )
        if not found_options:
            logger.debug("", current_stock_price, current_date, find_options_by)
            raise RolloverOptionNotFoundError("Not able to find next rollover option")
        else:
            next_option = found_options[0]
            return [
                Trade(
                    asset_type="option",
                    asset_id=next_option.id,
                    amount=amount,
                    date=current_date,
                ),
            ]

    return action


def close_option_position() -> RolloverAction:
    def action(
        listed_options: dict[str, Option], account: Account, current_date: pd.Timestamp
    ) -> list[Trade]:
        current_option = get_current_option(account)
        if current_option is not None:
            return [
                Trade(
                    asset_type="option",
                    asset_id=current_option.id,
                    amount=-current_option.position,
                    date=current_date,
                ),
            ]
        return []

    return action


# --- Strategy ---


def create_rollover_on_trade_fn(
    rollover_condition_action_pairs: list[tuple[RolloverCondition, RolloverAction]],
    default_action: RolloverAction | None = None,
) -> OnTradeFn:
    def on_trade(state: MarketState) -> list[Trade]:
        listed_options = state.listed_options
        account = state.account
        current_date = state.current_date

        current_option = get_current_option(account)
        trades: list[Trade] = []

        if current_option is None:
            # No current option, apply default action if provided
            if default_action is not None:
                trades.extend(default_action(listed_options, account, current_date))
        else:
            # Evaluate condition-action pairs
            for condition, action in rollover_condition_action_pairs:
                if condition(current_option, account, current_date):
                    trades.extend(action(listed_options, account, current_date))
                    break

        return trades

    return on_trade
