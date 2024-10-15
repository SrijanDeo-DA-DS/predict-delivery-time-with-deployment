from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import pandas as pd
import yaml

def load_params(params_path: str) -> float:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        n_estimators = params['model_building']['n_estimators']
        max_depth = params['model_building']['max_depth']
        logging.info(f"Successfully loaded n_estimators and max_depth from {params_path}.")
        return n_estimators, max_depth
    except FileNotFoundError:
        logging.error(f"File {params_path} not found.")
        raise
    except KeyError:
        logging.error("Key 'test_size' not found in the YAML file.")
        raise
    except yaml.YAMLError:
        logging.error(f"Failed to parse YAML file {params_path}.")
        raise

def train_model(X_train, y_train):
    """Train the Random Forest model."""
    rf = RandomForestRegressor(n_estimators, max_depth)
    rf.fit(X_train, y_train)
    return rf

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

def main():

    # Fetch the data from data/raw
    train_data = pd.read_csv('./data/interim/train_interim_transformed.csv')
    test_data = pd.read_csv('./data/interim/test_interim_transformed.csv')

    n_estimators = yaml.safe_load(open('params.yaml','r'))['model_building']['n_estimators']
    max_depth = yaml.safe_load(open('params.yaml','r'))['model_building']['max_depth']

    X_train = train_data.drop(labels=['Time_taken (min)'],axis=1)
    y_train = train_data[['Time_taken (min)']]

    rf = train_model(X_train, y_train)
        
    save_model(rf, 'models/model.pkl')

if __name__ == '__main__':
    main()