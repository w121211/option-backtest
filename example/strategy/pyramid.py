import os
from datetime import datetime, timedelta
from dataclasses import asdict, replace

import pandas as pd
import optuna
from optuna.trial import Trial


from option_backtest.bt_event import (
    BacktestConfig,
    GenerateOptionsConfig,
    GenerateStockConfig,
    run_backtest,
    calculate_metrics,
    plot_results,
)
from option_backtest.gen_data import generate_stock_data
from option_backtest.helper import generate_output_dir
from option_backtest.market import Account, OnTradeFn, Option, Trade
from option_backtest.strategy.buy_and_hold import buy_stock_on_day_0
from option_backtest.strategy.rollover_composed import (
    RolloverOptionNotFoundError,
    create_rollover_on_trade_fn,
    if_delta,
    if_expiration_within_days,
    rollover_to,
    buy_option,
)
from option_backtest.data.optionsdx import (
    preprocess_optionsdx_data,
    read_options_data,
    read_stock_data,
)


import optuna
from typing import Callable


def create_filtered_study(
    original_study: optuna.study.Study,
    storage_url: str,
    new_study_name: str,
    filter_function: Callable[[optuna.trial.FrozenTrial], bool],
) -> optuna.study.Study:
    """
    Filters trials from an original study and creates a new study in the specified database.

    Parameters:
    - original_study: The original optuna.study.Study object.
    - storage_url: The storage connection URL, e.g., "sqlite:///example.db".
    - new_study_name: The name of the new study, which must be unique within the database.
    - filter_function: A function that determines whether a trial should be included. It should accept a trial and return a boolean value.

    Returns:
    - A new study object.
    """
    # Create a new study with the same optimization direction as the original and connect it to the database
    study_filtered = optuna.create_study(
        study_name=new_study_name,
        direction=original_study.direction,
        storage=storage_url,
    )

    # Filter the trials from the original study using the provided filter function
    valid_trials = [t for t in original_study.get_trials() if filter_function(t)]

    # Add the filtered trials to the new study, only if they are complete
    for t in valid_trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            study_filtered.add_trial(t)

    return study_filtered


def buy_put_rollover() -> OnTradeFn:
    def rollover_to_delta(delta: float):
        return rollover_to(
            {
                "option_type": "put",
                # "dte_gt": 29,
                # "dte_lt": 61,
                "dte_gt": 1,
                "dte_lt": 30,
                "sort_by_delta_near": delta,
            },
            buy_amount=1,
        )

    condition_action_pairs = [
        # (if_delta(lambda x: x < -0.7), rollover_to_delta(-0.3)),
        # (if_delta(lambda x: x > -0.3), rollover_to_delta(-0.3)),
        # (if_stock_price_change(lambda x: x > 0.05), rollover_to_delta(0.4)),  # 單日大漲
        # (if_stock_price_change(lambda x: x < -0.05), rollover_to_delta(0.4)),  # 單日大跌
        (if_expiration_within_days(3), rollover_to_delta(-0.5)),
    ]
    default_action = buy_option(
        {
            "option_type": "put",
            # "dte_gt": 29,
            # "dte_lt": 61,
            "dte_gt": 1,
            "dte_lt": 30,
            "sort_by_delta_near": -0.5,
        },
        amount=1,
    )

    # strategy_stock_dynamic_position = create_stock_strateg()
    buy_put_rollover = create_rollover_on_trade_fn(
        condition_action_pairs,
        default_action,
    )

    return buy_put_rollover


def setup_generated_data_common() -> BacktestConfig:
    stock_id = "XYZ"
    base_output_dir = "./output"

    # Generate stock price
    gen_stock_config = GenerateStockConfig(
        s0=100.0,
        mu=0.05,
        sigma_stock=0.2,
        dt=1 / 252,
        n_steps=252,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
    )
    gen_options_config = GenerateOptionsConfig(
        r=0.05,
        sigma_option=0.2,
    )

    # Generate output path based on config params
    output_dir = generate_output_dir(
        asdict(gen_stock_config),
        base_output_dir,
        include_params=["s0", "mu", "sigma_stock"],
    )
    stock_price_path = f"{output_dir}/{stock_id}_price.csv"

    if not os.path.exists(stock_price_path):
        # Repeately use the generated stock prices if existed
        generate_stock_data(
            s0=gen_stock_config.s0,
            mu=gen_stock_config.mu,
            sigma=gen_stock_config.sigma_stock,
            dt=gen_stock_config.dt,
            start_date=gen_stock_config.start_date,
            end_date=gen_stock_config.end_date,
            output_path=stock_price_path,
        )

    # Set backtest config
    config = BacktestConfig(
        stock_id=stock_id,
        output_dir=output_dir,
        stock_price_path=stock_price_path,
        r=0.05,
        init_cash=1e5,
        gen_stock_config=gen_stock_config,
        gen_options_config=gen_options_config,
    )

    return config


def run_backtest_on_generated_data():
    config = setup_generated_data_common()

    # Run backtest for the protective put strategy
    on_trade_fns_protective_put = [
        buy_stock_on_day_0(amount=100),  # Buy stock on day 0 and hold
        buy_put_rollover(),
    ]
    (
        trades_protective_put,
        portfolio_df_protective_put,
        stock_price_df,
        listed_options,
    ) = run_backtest(config, on_trade_fns_protective_put)

    # Run backtest for the buy and hold strategy
    on_trade_fns_buy_and_hold = [buy_stock_on_day_0(amount=100)]
    trades_buy_and_hold, portfolio_df_buy_and_hold, _, _ = run_backtest(
        config,
        on_trade_fns_buy_and_hold,
        stock_price_df,
        listed_options,
    )

    # Calculate metrics for both strategies
    metrics_protective_put = calculate_metrics(portfolio_df_protective_put, 0)
    metrics_buy_and_hold = calculate_metrics(portfolio_df_buy_and_hold, 0)

    # Create a DataFrame to store the comparison metrics
    comparison_df = pd.concat(
        [metrics_protective_put, metrics_buy_and_hold],
        axis=0,
        keys=["Protective Put", "Buy and Hold"],
    )

    # Print the comparison DataFrame
    print("Comparison of Protective Put vs Buy and Hold:")
    print(comparison_df)

    # Save the comparison DataFrame to a CSV file
    comparison_df.to_csv(f"{config.output_dir}/comparison_metrics.csv", index=False)

    # Plot result
    plot_results(
        stock_price_df,
        listed_options,
        portfolio_df_protective_put,
        trades_protective_put,
    )


def monte_carlo_run():
    # num_simulations = 3
    # monte_carlo_results = run_monte_carlo_backtest(
    #     config,
    #     on_trade_fns,
    #     num_simulations,
    # )
    pass


def run_backtest_on_historical_data():
    stock_id = "NVDA"
    # data_path = "./data/optionsdx/nvda_eod_2023q1-0x56e5/nvda_eod_202301_mini.csv"
    # output_dir = "./data/optionsdx/processed/nvda_eod_202301_mini"
    data_path = "./data/optionsdx/nvda_eod_2023q1-0x56e5/nvda_eod_202301.txt"
    output_dir = "./data/optionsdx/processed/nvda_eod_202301"
    output_stock_csv = f"{output_dir}/NVDA_price.csv"
    output_options_csv = f"{output_dir}/NVDA_options.csv"

    # Prepare data
    if not os.path.exists(output_dir):
        preprocess_optionsdx_data(
            stock_id=stock_id,
            data_path=data_path,
            output_dir=output_dir,
        )
    stock_price_df = read_stock_data(output_stock_csv)
    options = read_options_data(output_options_csv)
    listed_options = {e.id: e for e in options}

    # Setup backtest
    start_date = stock_price_df.index.min()
    end_date = stock_price_df.index.max()

    config = BacktestConfig(
        stock_id=stock_id,
        output_dir="./output",
        start_date=start_date,
        end_date=end_date,
        r=0.00,
        init_cash=1e5,
    )

    # Run backtest for the protective put strategy
    on_trade_fns_protective_put = [
        buy_stock_on_day_0(amount=100),  # Buy stock on day 0 and hold
        buy_put_rollover(),
    ]
    (trades_protective_put, portfolio_df_protective_put, _, _) = run_backtest(
        config,
        on_trade_fns_protective_put,
        stock_price_df,
        listed_options,
    )

    # Run backtest for the buy and hold strategy
    on_trade_fns_buy_and_hold = [buy_stock_on_day_0(amount=100)]
    trades_buy_and_hold, portfolio_df_buy_and_hold, _, _ = run_backtest(
        config,
        on_trade_fns_buy_and_hold,
        stock_price_df,
        listed_options,
    )

    # Calculate metrics for both strategies
    metrics_protective_put = calculate_metrics(portfolio_df_protective_put, 0)
    metrics_buy_and_hold = calculate_metrics(portfolio_df_buy_and_hold, 0)

    # Create a DataFrame to store the comparison metrics
    comparison_df = pd.concat(
        [metrics_protective_put, metrics_buy_and_hold],
        axis=0,
        keys=["Protective Put", "Buy and Hold"],
    )

    # Print the comparison DataFrame
    print("Comparison of Protective Put vs Buy and Hold:")
    print(comparison_df)

    # Save the comparison DataFrame to a CSV file
    comparison_df.to_csv(f"{config.output_dir}/comparison_metrics.csv", index=False)

    # Plot result
    plot_results(
        stock_price_df,
        listed_options,
        portfolio_df_protective_put,
        trades_protective_put,
    )


if __name__ == "__main__":
    # single_run()
    # optimize_sharpe()
    run_backtest_on_historical_data()
