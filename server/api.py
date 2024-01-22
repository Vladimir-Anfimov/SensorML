from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
import pandas as pd
from io import StringIO
from fastapi.middleware.cors import CORSMiddleware
from generate_plots import PlotsGenerator

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
    user_df = pd.read_csv(StringIO(content.decode('utf-8')))
    user_df["Timestamp"] = pd.to_datetime(user_df["Timestamp"])
    return user_df

def get_prophet_images(plots):
    images = []
    for i, plot in enumerate(plots):
        image_path = f"./static/prophet_{i}.png"
        plot.savefig(image_path, bbox_inches="tight")
        images.append(f'{STATIC_PATH}/prophet_{i}.png')
    return images


def get_lstm_images(plots):
    images = []
    for i, plot in enumerate(plots):
        image_path = f"./static/lstm_{i}.png"
        plot.savefig(image_path, bbox_inches="tight")
        images.append(f'{STATIC_PATH}/lstm_{i}.png')
    return images


def get_seq2seq_images(plots):
    images = []
    for i, plot in enumerate(plots):
        image_path = f"./static/seq2seq_{i}.png"
        plot.savefig(image_path, bbox_inches="tight")
        images.append(f'{STATIC_PATH}/seq2seq_{i}.png')
    return images


@app.post("/diagnose")
async def upload_csv(file: UploadFile = File(...)):
    user_df = await get_df(file)
    
    plots_prophet, risk_df_prophet = ProphetPredictor.get_plots_and_df(user_df)
    prophet_result = {"images": get_prophet_images(plots_prophet), "risk": compute_risk(user_df, risk_df_prophet) }

    neural_net_generator = PlotsGenerator(user_df)
    
    plots_lstm, predicted_lstm = neural_net_generator.lstm_plots()
    lstm_result = {"images": get_lstm_images(plots_lstm), "risk": compute_risk(user_df, predicted_lstm) }
    
    plots_seq2seq, predicted_seq2seq = neural_net_generator.seq2seq_plots()
    seq2seq_result = {"images": get_seq2seq_images(plots_seq2seq), "risk": compute_risk(user_df, predicted_seq2seq) }
    
    return {
        "prophet": prophet_result,
        "lstm": lstm_result,
        "seq2seq": seq2seq_result,
    }
    
