import os
from datetime import datetime, timedelta
from dataclasses import asdict, replace

from option_backtest.bt_event import BacktestConfig, run_monte_carlo_backtest
from option_backtest.helper import generate_output_dir
from option_backtest.strategy.rollover_composed import (
    create_rollover_strategy,
    if_delta,
    if_expiration_within_days,
    rollover_to,
    open_option_position,
)


def backtest_strategy():
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

    condition_action_pairs = [
        (if_delta(lambda x: x > 0.7), rollover_to_delta(0.6)),
        (if_delta(lambda x: x < 0.3), rollover_to_delta(0.4)),
        # (if_stock_price_change(lambda x: x > 0.05), rollover_to_delta(0.4)),  # 單日大漲
        # (if_stock_price_change(lambda x: x < -0.05), rollover_to_delta(0.4)),  # 單日大跌
        (if_expiration_within_days(3), rollover_to_delta(0.5)),
    ]
    default_action = open_option_position(
        {
            "option_type": "put",
            "dte_gt": 29,
            "dte_lt": 61,
            "sort_by_delta_near": 0.5,
        },
        1,
    )

    # strategy_stock_dynamic_position = create_stock_strateg()
    strategy_rollover = create_rollover_strategy(
        condition_action_pairs,
        default_action,
    )
    strategy = [strategy_rollover]

    # Create the backtest configuration
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

    output_dir = generate_output_dir(
        asdict(config),
        config.output_dir,
        include_params=["s0", "mu", "sigma_stock"],
    )
    stock_price_dir = f"{output_dir}/stock_price"
    os.makedirs(stock_price_dir, exist_ok=True)

    config.output_dir = output_dir
    config.stock_price_dir = stock_price_dir

    # Run Monte Carlo simulations
    num_simulations = 3

    monte_carlo_results = run_monte_carlo_backtest(
        config,
        strategy,
        num_simulations,
    )

    # Print the Monte Carlo simulation results
    print("Monte Carlo Simulation Results:")
    print(monte_carlo_results)


if __name__ == "__main__":
    backtest_strategy()
