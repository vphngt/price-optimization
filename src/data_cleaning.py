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

# def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
#     """Extracts seasonal and calendar features from the date column."""
#     if 'date' not in df.columns:
#         raise ValueError("DataFrame must contain a 'date' column.")
        
#     df['year'] = df['date'].dt.year
#     df['month'] = df['date'].dt.month
#     df['day_of_week'] = df['date'].dt.dayofweek
#     df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
#     # return df

