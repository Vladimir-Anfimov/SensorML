from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
import pandas as pd
from io import StringIO
from fastapi.middleware.cors import CORSMiddleware

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

DAYS_OUTPUT = 28
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


@app.post("/diagnose")
async def upload_csv(file: UploadFile = File(...)):
    user_df = await get_df(file)
    
    plots, risk_df = ProphetPredictor.get_plots_and_df(user_df)

    print(risk_df)
    
    images = get_prophet_images(plots)

    risk = compute_risk(user_df, risk_df)
    print(risk)
    
    prophet_result = {"images": images, "risk": risk }
    
    return {
        "prophet": prophet_result,
    }
    
