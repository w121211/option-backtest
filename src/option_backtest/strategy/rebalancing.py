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

# --- Helpers ---

...

# --- Strategy ---


def create_stock_rebalancing_strategy() -> StrategyFn:
    def strategy(
        listed_options: dict[str, Option],
        account: Account,
        current_date: pd.Timestamp,
    ) -> list[Trade]:
        trades: list[Trade] = []

        ...

        return trades

    return strategy


# --- Example usage ---

if __name__ == "__main__":
    ...
