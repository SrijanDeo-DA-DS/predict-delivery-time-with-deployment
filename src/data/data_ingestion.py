import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

df=pd.read_csv('https://raw.githubusercontent.com/Shivan118/New-Machine-Learning-Modular-Coding-projecs/refs/heads/main/Data/finalTrain.csv')

train_data, test_data = train_test_split(df, test_size = 0.20, random_state= 42)

df.drop('ID',axis=1,inplace=True)

data_path='./data'

raw_data_path = os.path.join(data_path, 'raw')
os.makedirs(raw_data_path, exist_ok=True)

train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)