import pandas as pd


def clean_data(df):
    # reformat date
    df["date"] = pd.to_datetime(df["date"])
    # handle missing promotion values
    df["promotion_type"].fillna("None")
    # calculate lost margin
    df["potential_revenue"] = df["base_price"] * df["units_sold"]
    df["discount_leakage"] = df["potential_revenue"] - df["revenue"]
    return df


