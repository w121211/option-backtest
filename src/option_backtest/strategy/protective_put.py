import os
from datetime import datetime, timedelta
from dataclasses import asdict, replace

import pandas as pd

from ..bt_event import BacktestConfig, run_backtest
from ..gen_data import generate_stock_data
from ..helper import generate_output_dir
from ..market import Account, OnTradeFn, Option, Trade
from ..strategy.buy_and_hold import buy_stock_on_day_0
from ..strategy.rollover_composed import (
    create_rollover_on_trade_fn,
    if_delta,
    if_expiration_within_days,
    rollover_to,
    buy_option,
)


def protective_put() -> list[OnTradeFn]:
    def rollover_to_delta(delta: float):
        return rollover_to(
            {
                "option_type": "put",
                "dte_gt": 29,
                "dte_lt": 61,
                "sort_by_delta_near": delta,
            },
            buy_amount=1,
        )

    condition_action_pairs = [
        (if_delta(lambda x: x < -0.7), rollover_to_delta(-0.6)),
        (if_delta(lambda x: x > -0.3), rollover_to_delta(-0.4)),
        # (if_stock_price_change(lambda x: x > 0.05), rollover_to_delta(0.4)),  # 單日大漲
        # (if_stock_price_change(lambda x: x < -0.05), rollover_to_delta(0.4)),  # 單日大跌
        (if_expiration_within_days(3), rollover_to_delta(-0.5)),
    ]
    default_action = buy_option(
        {
            "option_type": "put",
            "dte_gt": 29,
            "dte_lt": 61,
            "sort_by_delta_near": -0.5,
        },
        amount=1,
    )

    # strategy_stock_dynamic_position = create_stock_strateg()
    buy_put_rollover = create_rollover_on_trade_fn(
        condition_action_pairs,
        default_action,
    )

    on_trade_fns = [
        buy_stock_on_day_0(amount=100),  # Buy stock on day 0 and hold
        buy_put_rollover,
    ]

    return on_trade_fns
