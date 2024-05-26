import copy
import os
import logging
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Literal, TypedDict
from typing_extensions import NotRequired

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..gen_data import generate_and_save_stock_data, generate_options
from ..market import Account, Option, Stock, StrategyFn, Trade, episode
from ..option import FindOptionsBy, find_options


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

FindOptionFilter = OrderedDict([])


@dataclass
class FindOptionFilterBy:
    delta_near: float | None = None


def select_next_option(
    listed_options: dict[str, Option],
    current_stock_price: float,
    current_date: pd.Timestamp,
    filter_by: FindOptionFilterBy,
) -> Option | None:
    found_options = find_options(
        listed_options,
        current_stock_price,
        current_date,
        {
            "option_type": "put",
            "dte_gt": 29,
            "dte_lt": 61,
            "sort_by_delta_near": filter_by.delta_near,
        },
    )
    if not found_options:
        return None
    return found_options[0]


def create_rollover_strategy(
    n_option_legs: int,
    entry_position: int,
    should_rollover_fn: Callable[[datetime, Option | None, Account], bool],
    select_next_option_fn: Callable[
        [dict[str, Option], float, pd.Timestamp, FindOptionFilterBy], Option | None
    ],
) -> StrategyFn:
    def rollover_strategy(
        listed_options: dict[str, Option],
        account: Account,
        current_date: pd.Timestamp,
    ) -> list[Trade]:
        # Validation
        if count_option_legs(account) > n_option_legs:
            raise Exception(
                f"This strategy supports up to {n_option_legs} option legs."
            )

        current_option = get_current_option(account)
        if should_rollover_fn(current_date, current_option, account):
            current_stock_price = account.stock.df.at[current_date, "price"]
            next_option = select_next_option_fn(
                listed_options,
                current_stock_price,
                current_date,
                FindOptionFilterBy(delta_near=0.5),
            )
            if next_option is None:
                logger.warning(
                    f"No suitable options found for rollover on {current_date}"
                )
                return []

            trades: list[Trade] = []

            # Exit the current option position
            if current_option is not None:
                trades.append(
                    Trade(
                        asset_type="option",
                        asset_id=current_option.id,
                        position=-current_option.position,
                        date=current_date,
                    )
                )

            # Enter the new option position
            trades.append(
                Trade(
                    asset_type="option",
                    asset_id=next_option.id,
                    position=entry_position,
                    date=current_date,
                )
            )

            next_option_delta = next_option.df.at[current_date, "delta"]
            logger.info(
                f"Rollover date={current_date.strftime('%y%m%d')}, price={current_stock_price:.2f}, delta={next_option_delta:.2f}, {current_option.id if current_option else 'None'} -> {next_option.id}"
            )
            return trades
        else:
            logger.info(f"No rollover needed on {current_date}")
            return []

    return rollover_strategy


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


def should_rollover(
    current_date: datetime, current_option: Option | None, account: Account
) -> bool:
    if current_option is None:
        return True

    dte = (current_option.expiry_date - current_date).days
    return dte < 2


# Example usage
if __name__ == "__main__":
    # Long call, rollover before 2 days, roll to ATM, 30 days+
    n_option_legs = 1
    entry_position = 1
    find_options_by: FindOptionsBy = {
        "option_type": "call",
        "dte_gt": 29,
        "dte_lt": 61,
        "sort_by_delta_near": 0.5,
    }
    rollover_criteria: ShouldRolloverCriteria = {"dte_lt": 2}
