# Enable inline dict typing. See https://github.com/microsoft/pyright/discussions/5682
# pyright: enableExperimentalFeatures=true

import copy
import os
import logging
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from typing import Callable, Literal, TypedDict
from typing_extensions import NotRequired

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .gen_data import generate_stock_data, generate_options
from .market import Account, Option, Stock, Trade, OnTradeFn, episode


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class GenerateStockConfig:
    # Generate stock prices
    s0: float = 100.0  # 初始價格
    mu: float = 0.05  # 預期收益率（年化）
    sigma_stock: float = 0.2  # 波動率（年化）
    dt: float = 1 / 252  # 時間步長（假設一年有252個交易日）
    n_steps: int = 252  # 模擬步數（假設模擬一年的價格路徑）


@dataclass
class GenerateOptionsConfig:
    # Generate option prices
    r: float = 0.05  # Risk-free rate
    sigma_option: float = 0.2  # Volatility


@dataclass
class BacktestConfig:
    # Basic
    stock_id: str = "XYZ"  # Stock ticker
    output_dir: str = "./output"  # Backtest output directory
    stock_price_dir: str | None = None  # Stock price directory for monte carlo backtest
    stock_price_path: str | None = None  # Stock price for backtest

    # Required params
    start_date: datetime = datetime(2023, 1, 1)
    end_date: datetime = datetime(2023, 12, 31)
    r: float = 0.05  # Risk-free rate for calculating Sharpe Ratio

    # Episode
    init_cash: float = 1e3

    # Configs for generating data
    gen_stock_config: GenerateStockConfig | None = None
    gen_options_config: GenerateOptionsConfig | None = None


def prepare_market_df(stock: Stock, options: list[Option]) -> pd.DataFrame:
    market_df = pd.DataFrame(index=stock.df.index)
    market_df[f"{stock.id}_price"] = stock.df["price"]

    for option in options:
        market_df[f"{option.id}_price"] = option.df["price"]

    return market_df


# --------- Process Backtest Result ---------


def calculate_metrics(
    portfolio_df: pd.DataFrame, risk_free_rate: float
) -> pd.DataFrame:
    returns = portfolio_df["value"].pct_change()  # 計算每日收益率
    total_return = (
        portfolio_df["value"].iloc[-1] / portfolio_df["value"].iloc[0]
    ) - 1  # 總回報率
    daily_risk_free_rate = risk_free_rate / 252  # 將年化無風險利率轉換為每日利率
    sharpe_ratio = (
        (returns.mean() - daily_risk_free_rate) / returns.std() * np.sqrt(252)
    )  # 計算年化Sharpe比率
    max_drawdown = (
        portfolio_df["value"] / portfolio_df["value"].cummax() - 1
    ).min()  # 最大回撤
    volatility = returns.std() * np.sqrt(252)  # 年化波動性
    std_dev = returns.std()  # 日波動性（標準差）

    metrics_data = {
        "Return": [total_return],
        "Sharpe": [sharpe_ratio],
        "MaxDrawdown": [max_drawdown],
        "Volatility": [volatility],
        "StdDev": [std_dev],  # 新增每日標準差
    }
    metrics_df = pd.DataFrame(metrics_data)
    return metrics_df


def save_outputs(
    config: BacktestConfig,
    trades: list[Trade],
    portfolio_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
):
    output_dir = config.output_dir
    trades_path = f"{output_dir}/trades.csv"
    portfolio_path = f"{output_dir}/portfolio.csv"
    metrics_path = f"{output_dir}/metrics.csv"

    pd.DataFrame(trades).to_csv(trades_path, index=False)
    portfolio_df.to_csv(portfolio_path)
    metrics_df.to_csv(metrics_path, index=False)


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
        if trade.asset_type == "option" and trade.amount > 0:
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
        if trade.asset_type == "option" and trade.amount < 0:
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


# --------- Backtest ---------


def run_backtest(
    config: BacktestConfig,
    on_trade_fns: list[OnTradeFn],
    stock_price_df: pd.DataFrame | None = None,
    listed_options: dict[str, Option] | None = None,
) -> tuple[
    list[Trade],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Option],
]:
    if stock_price_df is None:
        # Read stock price data from the path
        stock_price_df = pd.read_csv(config.stock_price_path)
        stock_price_df["date"] = pd.to_datetime(stock_price_df["date"])
        stock_price_df = stock_price_df.set_index("date")

    stock = Stock(id="XYZ", df=stock_price_df)

    if listed_options is None:
        if config.gen_options_config is not None:
            # Generate options data based on stock prices
            r = config.gen_options_config.r  # Risk-free rate
            sigma_option = config.gen_options_config.sigma_option  # Volatility

            strikes = np.arange(
                (min(stock.df["price"]) // 5 - 2) * 5,
                (max(stock.df["price"]) // 5 + 3) * 5 + 1,
                5,
            )
            expiry_dates = list(pd.date_range("2023-01-01", periods=104, freq="W"))
            listed_options = generate_options(
                stock,
                strikes,
                expiry_dates,
                r,
                sigma_option,
            )
        else:
            raise Exception("config.gen_options_config is None")

    # Run backtest
    trades, portfolio_df = episode(
        config.init_cash, stock, listed_options, on_trade_fns
    )

    return trades, portfolio_df, stock_price_df, listed_options


def run_monte_carlo_backtest(
    config: BacktestConfig,
    on_trade_fns: list[OnTradeFn],
    num_simulations: int,
) -> pd.DataFrame:
    """
    Run Monte Carlo simulations of the backtest.

    Parameters:
    - config: BacktestConfig object containing the backtest configuration.
    - strategies: List of trading strategies to be used in the backtest.
    - num_simulations: Number of Monte Carlo simulations to run.

    Returns:
    - A DataFrame containing the performance metrics for each simulation.
    """

    def get_stock_price_path(sim_index: int) -> str:
        return f"./{config.stock_price_dir}/{config.stock_id}_price_sim_{sim_index}.csv"

    performance_metrics = []

    # Generate stock data if not existed
    for i in range(num_simulations):
        stock_price_path = get_stock_price_path(i)
        if not os.path.exists(stock_price_path) and config.gen_stock_config is not None:
            generate_stock_data(
                s0=config.gen_stock_config.s0,
                mu=config.gen_stock_config.mu,
                sigma=config.gen_stock_config.sigma_stock,
                dt=config.gen_stock_config.dt,
                start_date=config.start_date,
                end_date=config.end_date,
                output_path=stock_price_path,
            )
        else:
            raise Exception("no stock_price_path & config.gen_stock_config is None")

    for i in range(num_simulations):
        print(f"Running simulation {i+1}/{num_simulations}")

        # Update the backtest configuration with the random parameters
        config_sim = copy.deepcopy(config)
        config_sim.stock_price_path = get_stock_price_path(i)

        # Create a unique output directory for each simulation
        config_sim.output_dir = f"{config.output_dir}/sim_{i+1}"
        os.makedirs(config_sim.output_dir, exist_ok=True)

        # Run backtest
        trades, portfolio_df, stock_price_df, listed_options = run_backtest(
            config_sim, on_trade_fns
        )

        # Extract performance metrics from backtest results
        metrics = calculate_metrics(portfolio_df, config.r)
        metrics["Simulation"] = i + 1
        performance_metrics.append(metrics)

    performance_df = pd.DataFrame(performance_metrics)

    # Save the performance metrics dataframe
    performance_path = f"{config.output_dir}/performance_metrics.csv"
    performance_df.to_csv(performance_path, index=False)

    return performance_df
