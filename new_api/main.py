import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse, Response, JSONResponse, StreamingResponse
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
from optimizer.optimization import SequenceOptimization
import io



app = FastAPI()
templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/upload", response_class=HTMLResponse)
async def upload_baseline_form(request: Request, response: Response):

    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/preprocess", response_class=JSONResponse)
async def preprocess(response: Response, file: UploadFile = File(...)):

    data = f"./uploaded_files/{file.filename}"
    with open(data, "wb") as f:
        f.write(file.file.read())

    events, labels = data_load(data)
    events.to_csv('preprocessed_data/events.csv', index=False)
    labels.to_csv('preprocessed_data/labels.csv', index=False)

    return {
        "message": "Done.'"
    }

@app.get("/model_train", response_class=JSONResponse)
async def model_train(request: Request):
    
    pbp_data = pd.read_csv('preprocessed_data/events.csv')
    labels = pd.read_csv('preprocessed_data/labels.csv')
    model_training(pbp_data, labels)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')  
    buf.seek(0) 

    return StreamingResponse(buf, media_type="image/png")


@app.get("/sequence_mining", response_class=HTMLResponse)
async def seq_min(request: Request):
    pbp_data = pd.read_csv('preprocessed_data/combined_df.csv')
    df = sequence_mining("home", "away", pbp_data, "DET")
    
    html_table = df.to_html(classes='table table-striped')
    
    return HTMLResponse(content=html_table)

@app.get("/optimization", response_class=HTMLResponse)
async def optimize(request: Request):

    opt = SequenceOptimization('runs_predictor.keras')
    df = opt.opt_loop('preprocessed_data/events.csv')

    html_table = df.to_html(classes='table table-striped')
    
    return HTMLResponse(content=html_table) 