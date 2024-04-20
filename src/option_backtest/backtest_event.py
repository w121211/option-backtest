# Enable inline dict typing. See https://github.com/microsoft/pyright/discussions/5682
# pyright: enableExperimentalFeatures=true

import copy
import os
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Literal, TypedDict
from typing_extensions import NotRequired

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .gen_data import generate_and_save_stock_data, generate_options
from .market import Account, Option, Stock, StrategyFunc, Trade, episode
from .option import FindOptionsBy, find_options

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def prepare_market_df(stock: Stock, options: list[Option]) -> pd.DataFrame:
    market_df = pd.DataFrame(index=stock.df.index)
    market_df[f"{stock.id}_price"] = stock.df["price"]

    for option in options:
        market_df[f"{option.id}_price"] = option.df["price"]

    return market_df


# def monte_carlo_demo(
#     sample_price_series: Callable,
#     generate_options: Callable,
#     strategy: Callable,
#     n_trials: int,
# ):
#     trial_portfolio_dfs = []

#     for i in range(n_trials):
#         price_series = sample_price_series()
#         stock = Stock(ticker="XYZ", df=price_series)
#         options = generate_options(price_series)
#         market_df = prepare_market_df(stock, options)

#         portfolio_df, trades = episode(market_df, strategy)
#         trial_portfolio_dfs.append(portfolio_df)

#     # Concatenate trial portfolio dataframes
#     combined_portfolio_df = pd.concat(trial_portfolio_dfs)

#     # Calculate metrics across trials
#     metrics = {
#         "Total Return": combined_portfolio_df["portfolio_value"].pct_change().mean(),
#         "Volatility": combined_portfolio_df["portfolio_value"].pct_change().std(),
#         # Add more metrics as needed
#     }

#     return metrics


def calculate_metrics(portfolio_df: pd.DataFrame, risk_free_rate: float = 0.02) -> dict:
    returns = portfolio_df["value"].pct_change()
    total_return = (portfolio_df["value"].iloc[-1] / portfolio_df["value"].iloc[0]) - 1
    sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)
    max_drawdown = (portfolio_df["value"] / portfolio_df["value"].cummax() - 1).min()

    metrics = {
        "Total Return": total_return,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Volatility": returns.std() * np.sqrt(252),
    }
    return metrics


def plot_results(
    stock_df: pd.DataFrame,
    listed_options: dict[str, Option],
    portfolio_df: pd.DataFrame,
    trades: list[Trade],
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=stock_df.index,
            y=stock_df["price"],
            name="Stock Price",
            line=dict(color="blue"),
            yaxis="y1",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df["value"],
            name="Portfolio Value",
            line=dict(color="green"),
            yaxis="y2",
        )
    )

    # Add entry trades as markers
    for trade in trades:
        if trade.asset_type == "option" and trade.position > 0:
            option = next(
                opt for opt in listed_options.values() if opt.id == trade.asset_id
            )
            fig.add_trace(
                go.Scatter(
                    x=[trade.date],
                    y=[option.strike],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=10,
                        color="red",
                        # line=dict(width=2, color="DarkSlateGrey"),
                    ),
                    name=f"Entry: {option.id}",
                    hovertext=f"Strike: {option.strike}<br>Expiry: {option.expiry_date.strftime('%y%m%d')}",
                    yaxis="y1",
                    legendgroup="markers",  # Set the same legendgroup for all markers
                    showlegend=False,  # Hide the markers from the legend
                )
            )
        if trade.asset_type == "option" and trade.position < 0:
            option = next(
                opt for opt in listed_options.values() if opt.id == trade.asset_id
            )
            fig.add_trace(
                go.Scatter(
                    x=[trade.date],
                    y=[option.strike],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=10,
                        color="darkred",
                        # line=dict(width=2, color="DarkSlateGrey"),
                    ),
                    name=f"Entry: {option.id}",
                    hovertext=f"Strike: {option.strike}<br>Expiry: {option.expiry_date.strftime('%y%m%d')}",
                    yaxis="y1",
                    legendgroup="markers",  # Set the same legendgroup for all markers
                    showlegend=False,  # Hide the markers from the legend
                )
            )

    fig.update_layout(
        title="Stock Price, Portfolio Value, and Entry Trades",
        # xaxis_title="Date",
        xaxis=dict(
            title="Date",
            tickformat="%y-%m-%d",  # Set the date format for the x-axis
        ),
        yaxis=dict(
            title="Stock Price",
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue"),
        ),
        yaxis2=dict(
            title="Portfolio Value",
            titlefont=dict(color="green"),
            tickfont=dict(color="green"),
            overlaying="y",
            side="right",
        ),
        legend=dict(x=0, y=1, orientation="h"),
    )

    fig.show()


# -------- Strategy --------


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


ShouldRolloverCriteria = TypedDict(
    "ShouldRolloverCriteria",
    {
        "dte_lt": NotRequired[int],
        "delta_gt": NotRequired[float],
        "delta_lt": NotRequired[float],
    },
)


def should_rollover(
    current_date: datetime,
    current_option: Option | None,
    criteria: ShouldRolloverCriteria,
) -> bool:
    # Example rollover criteria:
    # - If the current option has less than 10 days to expiry
    # - If the current option's delta is less than 0.3 or greater than 0.7
    if current_option is None:
        return True

    if criteria.get("dte_lt"):
        dte = (current_option.expiry_date - current_date).days
        return dte < criteria["dte_lt"]  # type: ignore

    return False

    # return (
    #     days_to_expiry < 10
    #     or current_option.df.loc[current_date, "delta"] < 0.3
    #     or current_option.df.loc[current_date, "delta"] > 0.7
    # )


def create_rollover_strategy(
    n_option_legs: int = 1,
    entry_position: int = -1,
    find_options_by: FindOptionsBy | None = None,
    rollover_criteria: ShouldRolloverCriteria | None = None,
) -> StrategyFunc:
    if find_options_by is None:
        find_options_by = {
            "option_type": "put",
            "dte_gt": 29,
            "dte_lt": 61,
            "sort_by_delta_near": -0.5,
        }
    if rollover_criteria is None:
        rollover_criteria = {"dte_lt": 2}

    def rollover_strategy(
        listed_options: dict[str, Option],
        account: Account,
        current_date: pd.Timestamp,
    ) -> list[Trade]:
        # Validation
        if count_option_legs(account) > n_option_legs:
            raise Exception("This strategy is one-leg only.")

        current_option = get_current_option(account)
        if should_rollover(current_date, current_option, criteria=rollover_criteria):
            current_stock_price = account.stock.df.at[current_date, "price"]
            found_options = find_options(
                listed_options, current_stock_price, current_date, find_options_by
            )
            if not found_options:
                logger.warning(
                    f"No suitable options found for rollover on {current_date}"
                )
                return []

            next_option = found_options[0]

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


def demo(strategy_fn: StrategyFunc):
    # Read or generate stock data
    stock_id = "XYZ"
    output_dir = f"output/{stock_id}"

    s0 = 100.0  # 初始價格
    mu = 0.05  # 預期收益率（年化）
    sigma = 0.2  # 波動率（年化）
    dt = 1 / 252  # 時間步長（假設一年有252個交易日）
    n_steps = 252  # 模擬步數（假設模擬一年的價格路徑）
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)

    stock_price_path = f"{output_dir}/{stock_id}_price.csv"
    if not os.path.exists(stock_price_path):
        generate_and_save_stock_data(
            s0, mu, sigma, dt, start_date, end_date, stock_price_path
        )

    stock_price_df = pd.read_csv(stock_price_path)
    stock_price_df["date"] = pd.to_datetime(stock_price_df["date"])
    stock_price_df = stock_price_df.set_index("date")
    stock = Stock(id="XYZ", df=stock_price_df)

    # Generate options data
    r = 0.05  # Risk-free rate
    sigma_option = 0.2  # Volatility

    strikes = np.arange(
        (min(stock.df["price"]) // 5 - 2) * 5,
        (max(stock.df["price"]) // 5 + 3) * 5 + 1,
        5,
    )
    expiry_dates = list(pd.date_range("2023-01-01", periods=13, freq="M"))
    listed_options = generate_options(
        stock,
        strikes,
        expiry_dates,
        r,
        sigma_option,
    )

    # Run backtest
    init_cash = 1e3
    trades, portfolio_df = episode(init_cash, stock, listed_options, strategy_fn)

    # Calculate metrics
    metrics = calculate_metrics(portfolio_df)

    print("Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Plot results
    plot_results(stock.df, listed_options, portfolio_df, trades)


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

    strategy = create_rollover_strategy(
        n_option_legs,
        entry_position,
        find_options_by,
        rollover_criteria,
    )
    demo(strategy)
