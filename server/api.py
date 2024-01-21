from fastapi import FastAPI, File, UploadFile
import pandas as pd
from io import StringIO
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = FastAPI()

@app.post("/diagnose")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode('utf-8')))
    
    print(df.head())
    return {"status": "ok"}
