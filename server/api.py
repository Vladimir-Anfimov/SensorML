from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
import pandas as pd
from io import StringIO
from fastapi.middleware.cors import CORSMiddleware
from generate_plots import PlotsGenerator
from copy import deepcopy

from prophet_train import ProphetPredictor
from risk import compute_risk

app = FastAPI()


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="./static"), name="static")

STATIC_PATH = "http://localhost:8000/static"


async def get_df(file):
    content = await file.read()
    user_df = pd.read_csv(StringIO(content.decode("utf-8")))
    user_df["Timestamp"] = pd.to_datetime(user_df["Timestamp"])
    return user_df


def get_prophet_images(plots):
    images = []
    for i, plot in enumerate(plots):
        image_path = f"./static/prophet_{i}.png"
        plot.savefig(image_path, bbox_inches="tight")
        images.append(f"{STATIC_PATH}/prophet_{i}.png")
    return images


def get_lstm_images(plots):
    images = []
    for i, plot in enumerate(plots):
        image_path = f"./static/lstm_{i}.png"
        plot.savefig(image_path, bbox_inches="tight")
        images.append(f"{STATIC_PATH}/lstm_{i}.png")
    return images


def get_seq2seq_images(plots):
    images = []
    for i, plot in enumerate(plots):
        image_path = f"./static/seq2seq_{i}.png"
        plot.savefig(image_path, bbox_inches="tight")
        images.append(f"{STATIC_PATH}/seq2seq_{i}.png")
    return images


@app.post("/diagnose")
async def upload_csv(file: UploadFile = File(...)):
    user_df = await get_df(file)

    plots_prophet, risk_df_prophet = ProphetPredictor.get_plots_and_df(user_df)
    prophet_result = {
        "images": get_prophet_images(plots_prophet),
    }

    neural_net_generator = PlotsGenerator(user_df)

    plots_lstm, predicted_lstm = neural_net_generator.lstm_plots()
    lstm_result = {
        "images": get_lstm_images(plots_lstm),
    }

    plots_seq2seq, predicted_seq2seq = neural_net_generator.seq2seq_plots()
    seq2seq_result = {
        "images": get_seq2seq_images(plots_seq2seq),
    }

    prophet_risk = compute_risk(user_df, risk_df_prophet)
    lstm_risk = compute_risk(user_df, predicted_lstm)
    seq2seq_risk = compute_risk(user_df, predicted_seq2seq)
    overall_risk = deepcopy(prophet_risk)
    for i in range(len(overall_risk)):
        overall_risk[i] = (
            overall_risk[i][0],
            (prophet_risk[i][1] + lstm_risk[i][1] + seq2seq_risk[i][1]) // 3,
        )

    risks = []
    if len(prophet_risk) != len(lstm_risk) or len(lstm_risk) != len(
        seq2seq_risk
    ):
        print("ERROR: Different number of diseases in risk computation")

    for i in range(len(prophet_risk)):
        risks.append(
            {
                "disease": prophet_risk[i][0],
                "prophet": prophet_risk[i][1],
                "lstm": lstm_risk[i][1],
                "seq2seq": seq2seq_risk[i][1],
                "overall": overall_risk[i][1],
            }
        )

    print(risks)

    return {
        "prophet": prophet_result,
        "lstm": lstm_result,
        "seq2seq": seq2seq_result,
        "risks": risks,
    }
