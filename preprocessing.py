import pandas as pd
import preprocessing_functions as pf

df = pd.read_csv('path/to/dataframe.csv')

train = pf.Preprocessing(df)
train.preprocess()