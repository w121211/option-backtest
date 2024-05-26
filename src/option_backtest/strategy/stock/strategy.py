"""
@TODO
1. Change condition to signal, signal can be divided into 
    1. static signal: a dataframe, calculated at the beggining.
    2. event signal: a callable function and return boolean, need to be computed on trade.
2. Rename strategy as to_trade or on_trade?

```
signal_action_pairs = [
    (sma_above(), do_something()),
    (sma_below(), ...)
    (sma_crosses_price_above(), ...)
    (sma_crosses_price_below(), ...)
    (event_signal(), ...)
]
```

"""

from datetime import timedelta
from typing import Callable

import pandas as pd

from ...market import Account, Option, Stock, Trade, TradeActionFn


def create_stock_strategy(
    # preprocess_stock_df: list[Callable[...]],
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
