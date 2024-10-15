from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import pandas as pd

def evaluate_reg(true, predicted):
    r2 = r2_score(true, predicted) # Calculate r2 score
    MAE = mean_absolute_error(true, predicted) # Calculate MAE
    MSE = mean_squared_error(true, predicted) # Calculate MSE
    rmse = np.sqrt(mean_squared_error(true, predicted))
   
    return r2, MAE , MSE,rmse

def train_model(X_train, y_train):
    """Train the Random Forest model."""
    rf = RandomForestRegressor()
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

    X_train = train_data.drop(labels=['Time_taken (min)'],axis=1)
    y_train = train_data[['Time_taken (min)']]

    rf = train_model(X_train, y_train)
        
    save_model(rf, 'models/model.pkl')

if __name__ == '__main__':
    main()