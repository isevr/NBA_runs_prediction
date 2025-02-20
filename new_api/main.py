import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import time
from uuid import uuid4
from fastapi.staticfiles import StaticFiles
from model.preprocessing import data_load
from model.model_training import model_training
from sequence_mining.sequence_mining import sequence_mining



app = FastAPI()
templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/preprocess", response_class=HTMLResponse)
async def preprocess(request: Request, data: str = Form(...)):
    preprocessed_data = data_load(data)
    preprocessed_data.to_csv('pp_data.csv', index=False)

@app.post("/model_train", response_class=HTMLResponse)
async def model_train(request: Request):
    pbp_data = pd.read_csv('pp_data.csv')
    model_training(pbp_data[0], pbp_data[1])

    return templates.TemplateResponse("model_train.html", {
        "request": request,
        "training_result": "Model training completed successfully."  
    })

@app.post("/sequence_mining", response_class=HTMLResponse)
async def sequence_mining(request: Request):
    pbp_data = pd.read_csv('pp_data.csv')
    df = sequence_mining("home", "away", pbp_data, "DET","NBA_PBP_2015-16.csv")