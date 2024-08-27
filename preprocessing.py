import pandas as pd
import preprocessing_functions as pf

def data_load(data_path):
    df = pd.read_csv(data_path)

    train = pf.Preprocessing(df)
    
    return train.preprocess()