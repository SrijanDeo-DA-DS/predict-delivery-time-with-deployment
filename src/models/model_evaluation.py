import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_model(file_path: str):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

def evaluate_model(rf, X_test, y_test):

    y_pred = rf.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics_dict = {
            'mean_absolute_error':mae, 
            'mean_squared_error':mse, 
            'r2_score':r2,
            'root_mse':np.sqrt(mse)
        }
    return metrics_dict

def save_metrics(metrics, file_path):
    with open(file_path, 'w') as file:
        json.dump(metrics, file, indent=4)


def main():
    try:
        rf = load_model('./models/model.pkl')

        test_data = pd.read_csv('./data/interim/test_interim_transformed.csv')
        
        X_test = test_data.drop(['Time_taken (min)'], axis=1)
        y_test = test_data[['Time_taken (min)']]

        metrics = evaluate_model(rf, X_test, y_test)
        
        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()