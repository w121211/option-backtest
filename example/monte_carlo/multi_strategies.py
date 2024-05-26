import os
from datetime import datetime, timedelta
from dataclasses import asdict, replace

import pandas as pd

from option_backtest.bt_event import BacktestConfig, run_monte_carlo_backtest
from option_backtest.helper import generate_output_dir
from option_backtest.strategy.rollover_composed import (
    create_rollover_strategy,
    if_delta,
    if_expiration_within_days,
    rollover_to,
    open_option_position,
)


def backtest_multi_strategies():
    num_simulations = 3
    config = BacktestConfig(
        stock_id="XYZ",
        output_dir="./output/monte_carlo",
        s0=100.0,
        mu=0.05,
        sigma_stock=0.2,
        dt=1 / 252,
        n_steps=252,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        r=0.05,
        sigma_option=0.2,
        init_cash=1e3,
    )

    # Setup market conditions
    market_settings = [
        {"sigma_stock": 0.2, "sigma_option": 0.2},
        {"sigma_stock": 0.5, "sigma_option": 0.5},
        {"sigma_stock": 1.0, "sigma_option": 1.0},
    ]

    # Setup strategies
    def rollover_to_delta(delta: float):
        return rollover_to(
            {
                "option_type": "call",
                "dte_gt": 29,
                "dte_lt": 61,
                "sort_by_delta_near": delta,
            },
            1,
        )

    default_action = open_option_position(
        {
            "option_type": "put",
            "dte_gt": 29,
            "dte_lt": 61,
            "sort_by_delta_near": 0.5,
        },
        1,
    )

    rollover_naive = [(if_expiration_within_days(3), rollover_to_delta(0.5))]
    rollover_dynamic = [
        (if_delta(lambda x: x > 0.7), rollover_to_delta(0.6)),
        (if_delta(lambda x: x < 0.3), rollover_to_delta(0.4)),
        # (if_stock_price_change(lambda x: x > 0.05), rollover_to_delta(0.4)),  # 單日大漲
        # (if_stock_price_change(lambda x: x < -0.05), rollover_to_delta(0.4)),  # 單日大跌
        (if_expiration_within_days(3), rollover_to_delta(0.5)),
    ]
    rollover_naive = create_rollover_strategy(rollover_naive, default_action)
    rollover_dynamic = create_rollover_strategy(rollover_dynamic, default_action)

    strategy_configs = [
        {"name": "rollover_naive", "strategy": [rollover_naive]},
        {"name": "rollover_dynamic", "strategy": [rollover_dynamic]},
    ]

    # Run backtest
    performance_metrics = []

    for market_setting in market_settings:
        for i, strategy_config in enumerate(strategy_configs):
            print(
                f"Running Monte Carlo simulations for strategy {strategy_config['name']} and market setting {market_setting}"
            )

            # Update the backtest configuration with the market settings
            output_market_dir = generate_output_dir(
                asdict(config),
                config.output_dir,
                include_params=["s0", "mu", "sigma_stock"],
            )
            output_market_strategy_dir = (
                f"{output_market_dir}/{strategy_config['name']}/"
            )
            stock_price_dir = f"{output_market_dir}/stock_price"
            os.makedirs(output_market_strategy_dir, exist_ok=True)
            os.makedirs(stock_price_dir, exist_ok=True)

            config_sim = replace(
                config,
                output_dir=output_market_strategy_dir,
                stock_price_dir=stock_price_dir,
                **market_setting,
            )

            performance_df = run_monte_carlo_backtest(
                config_sim,
                strategy_config["strategy"],
                num_simulations,
            )

            # Merge performance metrics
            merged_metrics = {
                "Strategy": strategy_config["name"],
                # "mu": market_setting["mu"],
                "sigma_stock": market_setting["sigma_stock"],
                "Total Return Mean": performance_df["Total Return"].mean(),
                "Total Return Std": performance_df["Total Return"].std(),
                "Sharpe Ratio Mean": performance_df["Sharpe Ratio"].mean(),
                "Sharpe Ratio Std": performance_df["Sharpe Ratio"].std(),
                "Max Drawdown Mean": performance_df["Max Drawdown"].mean(),
                "Max Drawdown Std": performance_df["Max Drawdown"].std(),
                "Max Drawdown Abs": performance_df["Max Drawdown"].abs().max(),
                "Volatility Mean": performance_df["Volatility"].mean(),
                "Volatility Std": performance_df["Volatility"].std(),
            }
            performance_metrics.append(merged_metrics)

    merged_performance_df = pd.DataFrame(performance_metrics)

    # Save the merged performance metrics dataframe
    merged_performance_path = f"{config.output_dir}/merged_performance_metrics.csv"
    merged_performance_df.to_csv(merged_performance_path, index=False)

    return merged_performance_df


if __name__ == "__main__":
    backtest_multi_strategies()
