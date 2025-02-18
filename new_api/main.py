import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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



app = FastAPI()
templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/model_train", response_class=HTMLResponse)
async def model_train(request: Request, data: str = Form(...)):
    pbp_data = data_load(data)  
    model_training(pbp_data[0], pbp_data[1])

    return templates.TemplateResponse("model_train.html", {
        "request": request,
        "training_result": "Model training completed successfully."  
    })