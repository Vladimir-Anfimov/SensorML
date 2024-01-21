from data_frames import FrameLoader
import pandas as pd


def compute_risk(
    given_df: pd.DataFrame, forecast_df: pd.DataFrame
) -> list[tuple]:
    diseases = [
        {
            "name": "EarlyBlight",
            "lowAir": 24,
            "highAir": 29,
            "lowHum": 90,
            "highHum": 100,
        },
        {
            "name": "LateBlight",
            "lowAir": 10,
            "highAir": 24,
            "lowHum": 90,
            "highHum": 100,
        },
        {
            "name": "LeafMold",
            "lowAir": 21,
            "highAir": 24,
            "lowHum": 85,
            "highHum": 100,
        },
        {
            "name": "GrayMold",
            "lowAir": 17,
            "highAir": 23,
            "lowHum": 90,
            "highHum": 100,
        },
        {
            "name": "PowderyMildew",
            "lowAir": 22,
            "highAir": 30,
            "lowHum": 50,
            "highHum": 75,
        },
    ]

    ans = []
    for disease in diseases:
        ans.append(
            compute_risk_disease(
                given_df,
                forecast_df,
                disease["name"],
                disease["lowAir"],
                disease["highAir"],
                disease["lowHum"],
                disease["highHum"],
            )
        )

    print(ans)
    return ans


def compute_risk_disease(
    given_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    disease: str,
    lowAir: int,
    highAir: int,
    lowHum: int,
    highHum: int,
) -> tuple[str, int]:
    temp1 = compute_risk_column(given_df, forecast_df, "temp1", lowAir, highAir)
    temp2 = compute_risk_column(given_df, forecast_df, "temp2", lowAir, highAir)
    umid = compute_risk_column(given_df, forecast_df, "umid", lowHum, highHum)

    bad = temp1[0] + temp2[0] + umid[0]
    total = temp1[1] + temp2[1] + umid[1]

    return disease, bad * 100 // total


def compute_risk_column(
    given_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    column: str,
    low: int,
    high: int,
) -> tuple[int, int]:
    known_risk = compute_risk_helper(given_df, column, low, high)
    forecast_risk = compute_risk_helper(forecast_df, column, low, high)

    bad = known_risk[0] + forecast_risk[0]
    total = known_risk[1] + forecast_risk[1]

    return bad, total


def compute_risk_helper(
    df: pd.DataFrame, column: str, low: int, high: int
) -> tuple[int, int]:
    return len(df[(df["temp1"] >= low) & (df["temp1"] <= high)]), len(df)


df = FrameLoader(FrameLoader.RAW).load()
cutoff = df.shape[0] // 2
uploaded_df = df.iloc[:cutoff]
forecast_df = df.iloc[cutoff:]

compute_risk(uploaded_df, forecast_df)
