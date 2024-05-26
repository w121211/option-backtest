from datetime import datetime, timedelta
from typing import Callable, Literal, TypedDict
from typing_extensions import NotRequired
import pandas as pd

from .market import Option


FindOptionsBy = TypedDict(
    "FindOptionsBy",
    {
        # Filter options by the "call" or "put".
        "option_type": Literal["call", "put"],
        # Filter options by the "days to expiration" greter than x days.
        "dte_gt": NotRequired[int],
        # Filter options by the "days to expiration" less than x days.
        "dte_lt": NotRequired[int],
        # Filter options by the delta is greater than x. Use this to filter ITM options by delta > 0.5.
        "delta_gt": NotRequired[float],
        # Filter options by the "days to expiration" within a range (min_days, max_days).
        "dte_range": NotRequired[tuple[int, int]],
        # Filter options by the delta is less than x. Use this to filter OTM options by delta < 0.5.
        "delta_lt": NotRequired[float],
        # Sort by delta near x
        "sort_by_delta_near": NotRequired[float],
    },
)


def find_options(
    listed_options: dict[str, Option],
    current_stock_price: float,
    current_date: pd.Timestamp,
    attrs: FindOptionsBy,
) -> list[Option]:
    filtered_options = list(listed_options.values())

    if attrs.get("option_type"):
        filtered_options = [
            op for op in filtered_options if op.option_type == attrs["option_type"]
        ]

    if attrs.get("dte_gt"):
        dte_gt = attrs["dte_gt"]  # type: ignore
        filtered_options = [
            op
            for op in filtered_options
            if (op.expiry_date - current_date).days > dte_gt
        ]

    if attrs.get("dte_lt"):
        dte_lt = attrs["dte_lt"]  # type: ignore
        filtered_options = [
            op
            for op in filtered_options
            if (op.expiry_date - current_date).days < dte_lt
        ]

    if attrs.get("dte_range"):
        min_days, max_days = attrs["dte_range"]  # type: ignore
        filtered_options = [
            op
            for op in filtered_options
            if min_days <= (op.expiry_date - current_date).days < max_days
        ]

    if attrs.get("delta_gt"):
        delta_gt = attrs["delta_gt"]  # type: ignore
        filtered_options = [
            op for op in filtered_options if op.df.at[current_date, "delta"] > delta_gt
        ]
        filtered_options = sorted(
            filtered_options, key=lambda op: op.df.at[current_date, "delta"]
        )

    if attrs.get("delta_lt"):
        delta_lt = attrs["delta_lt"]  # type: ignore
        filtered_options = [
            op for op in filtered_options if op.df.at[current_date, "delta"] < delta_lt
        ]
        filtered_options = sorted(
            filtered_options,
            key=lambda op: op.df.at[current_date, "delta"],
            reverse=True,
        )

    # Sort options by delta near the specified value
    if attrs.get("sort_by_delta_near"):
        delta_near = attrs["sort_by_delta_near"]  # type: ignore

        def sort_key(op):
            try:
                return abs(op.df.at[current_date, "delta"] - delta_near)
            except KeyError:
                return float("inf")

        filtered_options = sorted(filtered_options, key=sort_key)

    return filtered_options


# FindOptionsCondition = tuple[
#     Literal["delta", "dte", "option_type"], Callable[[float | str], bool]
# ]


# def find_options(
#     listed_options: dict[str, Option],
#     current_stock_price: float,
#     current_date: pd.Timestamp,
#     conditions: list[FindOptionsCondition],
#     sort_by_delta_near: float | None = None,
# ) -> list[Option]:
#     def apply_conditions(op: Option) -> bool:
#         for cond_type, cond_func in conditions:
#             if cond_type == "delta":
#                 if not cond_func(op.df.at[current_date, "delta"]):
#                     return False
#             elif cond_type == "dte":
#                 dte = (op.expiry_date - current_date).days
#                 if not cond_func(dte):
#                     return False
#             elif cond_type == "option_type":
#                 if not cond_func(op.option_type):
#                     return False
#         return True

#     filtered_options = [op for op in listed_options.values() if apply_conditions(op)]

#     if sort_by_delta_near is not None:
#         filtered_options = sorted(
#             filtered_options,
#             key=lambda op: abs(op.df.at[current_date, "delta"] - sort_by_delta_near),
#         )

#     return filtered_options


def should_rollover(option: Option, current_date: datetime) -> bool:
    """Determine if the held option should be rolled over based on the current date."""
    return (option.expiry_date - current_date).days <= 10


def find_next_option(
    available_options: list[Option], current_date: datetime, current_market_price: float
) -> Option | None:
    """Find the next option to rollover to based on the criteria:
    1. Has at least 30 days before expiry.
    2. Is nearest to the current day.
    3. Is ATM (At-The-Money).
    """
    # Filter options that have at least 30 days before expiry
    valid_options = [
        option
        for option in available_options
        if (option.expiry_date - current_date).days >= 30
    ]
    if not valid_options:
        return None

    # Finding the option that is nearest to being At-The-Money (ATM)
    atm_option = min(
        valid_options,
        key=lambda x: (
            abs(x.strike - current_market_price),
            (x.expiry_date - current_date).days,
        ),
    )
    return atm_option


def rollover_options(
    available_options: list[Option], prices_df: pd.DataFrame
) -> list[Option]:
    prices_df["date"] = pd.to_datetime(prices_df["date"])

    rolled_over_options = []
    current_option = None

    for i, row in prices_df.iterrows():
        current_date = row["date"]
        current_market_price = row["price"]

        if current_option is None or should_rollover(current_option, current_date):
            next_option = find_next_option(
                available_options, current_date, current_market_price
            )
            if next_option and (
                current_option is None or next_option.id != current_option.id
            ):
                # if current_option:
                #     current_option.exit_date = current_date
                # next_option.entry_date = current_date

                rolled_over_options.append(next_option)
                print(
                    f"{current_date.strftime('%y-%m-%d')} {current_market_price} => Rolled over to option {next_option.strike}@{next_option.expiry_date.strftime('%y%m%d')}"
                )
                current_option = next_option  # Update the held option
            elif not next_option:
                print("No suitable option found to rollover to.")
                break  # Exit the loop if no suitable next option is found

    return rolled_over_options


def rollover_short_atm_calls(
    available_options: list[Option], prices_df: pd.DataFrame
) -> list[Option]:
    """
    Implement a short ATM (At-The-Money) call option rollover strategy.

    The strategy sells the nearest ATM call option with at least 30 days to expiry
    and rolls over to the next ATM call option when there are 10 days or fewer left to expiry.

    Parameters:
    - available_options: List of available Option objects to choose from.
    - prices_df: DataFrame with columns ['date', 'price'] representing the underlying asset prices.

    Returns:
    - List of Option objects representing the rolled-over short call options.
    """
    prices_df["date"] = pd.to_datetime(prices_df["date"])

    rolled_over_options = []
    current_option = None

    for i, row in prices_df.iterrows():
        current_date = row["date"]
        current_market_price = row["price"]

        if (
            current_option is None
            or (current_option.expiry_date - current_date).days <= 10
        ):
            # Find the next ATM call option to sell
            valid_options = [
                option
                for option in available_options
                if (option.expiry_date - current_date).days >= 30
                and option.option_type == "call"
            ]
            if valid_options:
                next_option = min(
                    valid_options,
                    key=lambda x: (
                        abs(x.strike - current_market_price),
                        (x.expiry_date - current_date).days,
                    ),
                )
                if current_option is None or next_option.id != current_option.id:
                    # if current_option:
                    #     current_option.exit_date = current_date
                    # current_option.position = 0  # Close the current short position
                    # next_option.entry_date = current_date
                    next_option.position = -1  # Open a new short position

                    rolled_over_options.append(next_option)
                    print(
                        f"{current_date.strftime('%y-%m-%d')} {current_market_price} => Rolled over to selling option {next_option.strike}@{next_option.expiry_date.strftime('%y%m%d')}"
                    )
                    current_option = next_option  # Update the sold option
            else:
                print("No suitable option found to rollover to.")
                break  # Exit the loop if no suitable next option is found

    return rolled_over_options
