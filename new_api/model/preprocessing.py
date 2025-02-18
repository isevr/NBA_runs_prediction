import pandas as pd
import model.preprocessing_functions as pf

def data_load(data_path):
    df = pd.read_csv(data_path)

    train = pf.Preprocessing(df)
    
    return train.preprocess()