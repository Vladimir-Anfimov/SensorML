from data_frames import FrameLoader
import pandas as pd


def compute_risk(
    given_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    column: str,
    low: int,
    high: int,
):
    temp1_risk = compute_risk_helper(given_df, column, low, high)
    temp2_risk = compute_risk_helper(given_df, "temp2", low, high)
    umid_risk = compute_risk_helper(given_df, "umid", low, high)

    temp1_forecast_risk = compute_risk_helper(forecast_df, "temp1", low, high)
    temp2_forecast_risk = compute_risk_helper(forecaste, "temp2", low, high)
    umid_forecast_risk = compute_risk_helper(forecast_df, "umid", low, high)


def compute_risk_helper(
    df: pd.DataFrame, column: str, low: int, high: int
) -> tuple[int, int]:
    return len(df[(df["temp1"] >= low) & (df["temp1"] <= high)]), len(df)


df = FrameLoader(FrameLoader.RAW).load()
compute_risk(df, df)
