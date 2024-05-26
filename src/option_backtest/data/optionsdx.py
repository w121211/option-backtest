import os

import pandas as pd

from ..market import Option
from ..gen_data import get_option_id, parse_option_id


def preprocess_optionsdx_data(stock_id: str, data_path: str, output_dir: str):
    if os.path.exists(output_dir):
        raise Exception("os.path.exists(output_dir)")

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip("[] ")

    df["date"] = pd.to_datetime(df["QUOTE_DATE"])
    df["EXPIRE_DATE"] = pd.to_datetime(df["EXPIRE_DATE"])

    # Create stock price DataFrame from the underlying stock price
    stock_price_df = df[["date", "UNDERLYING_LAST"]].drop_duplicates()
    stock_price_df.set_index("date", inplace=True)
    stock_price_df.columns = ["price"]

    # Split the data into separate DataFrames for calls and puts
    call_df = df[
        [
            "date",
            "UNDERLYING_LAST",
            "EXPIRE_DATE",
            "C_DELTA",
            "C_GAMMA",
            "C_VEGA",
            "C_THETA",
            "C_RHO",
            "C_IV",
            "C_VOLUME",
            "C_LAST",
            "C_SIZE",
            "C_BID",
            "C_ASK",
            "STRIKE",
        ]
    ]
    put_df = df[
        [
            "date",
            "UNDERLYING_LAST",
            "EXPIRE_DATE",
            "P_DELTA",
            "P_GAMMA",
            "P_VEGA",
            "P_THETA",
            "P_RHO",
            "P_IV",
            "P_VOLUME",
            "P_LAST",
            "P_SIZE",
            "P_BID",
            "P_ASK",
            "STRIKE",
        ]
    ]

    # Group the data by expiry date and strike for calls and puts
    call_grouped = call_df.groupby(["EXPIRE_DATE", "STRIKE"])
    put_grouped = put_df.groupby(["EXPIRE_DATE", "STRIKE"])

    # Create a list to store the option DataFrames
    option_dfs = []

    for (expiry_date, strike), group in call_grouped:
        option_id = get_option_id(
            stock_id,
            strike=strike,
            expiry_date=expiry_date,
            option_type="call",
        )
        # option_df = group[["date", "C_LAST", "C_DELTA", "C_VEGA"]]
        option_df = group.loc[:, ["date", "C_LAST", "C_DELTA", "C_VEGA"]].copy()
        option_df.columns = ["date", "price", "delta", "vega"]
        option_df.insert(0, "option_id", option_id)
        option_dfs.append(option_df)

    for (expiry_date, strike), group in put_grouped:
        option_id = get_option_id(
            stock_id,
            strike=strike,
            expiry_date=expiry_date,
            option_type="put",
        )
        # option_df = group[["date", "P_LAST", "P_DELTA", "P_VEGA"]]
        option_df = group.loc[:, ["date", "P_LAST", "P_DELTA", "P_VEGA"]].copy()
        option_df.columns = ["date", "price", "delta", "vega"]
        option_df.insert(0, "option_id", option_id)
        option_dfs.append(option_df)

    # Concatenate all option DataFrames into a single DataFrame
    all_options_df = pd.concat(option_dfs)

    # ------ Save To Output Directory ------
    os.makedirs(output_dir, exist_ok=True)

    # Save stock price data as CSV
    stock_price_file = os.path.join(output_dir, f"{stock_id}_price.csv")
    stock_price_df.to_csv(stock_price_file)

    # Save all options data as a single CSV file
    all_options_file = os.path.join(output_dir, f"{stock_id}_options.csv")
    all_options_df.to_csv(all_options_file, index=False)


def read_options_data(all_options_file: str) -> list[Option]:
    # Read the all_options_file and create a list of standalone option DataFrames
    all_options_df = pd.read_csv(all_options_file)
    all_options_df["date"] = pd.to_datetime(all_options_df["date"])

    options: list[Option] = []
    for option_id, group in all_options_df.groupby("option_id"):
        option_df = group[["date", "price", "delta", "vega"]].set_index("date")
        stock_id, expiry_date, option_type, strike = parse_option_id(option_id)  # type: ignore
        option = Option(
            id=option_id,  # type: ignore
            option_type=option_type,
            strike=strike,
            expiry_date=pd.to_datetime(expiry_date),
            df=option_df,
        )
        options.append(option)

    return options


def read_stock_data(stock_price_file: str) -> pd.DataFrame:
    # Read the stock price CSV file
    stock_price_df = pd.read_csv(stock_price_file, index_col="date", parse_dates=True)
    return stock_price_df
