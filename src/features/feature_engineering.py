from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
import math
import os


def create_preprocessor(numerical_columns, categorical_columns, ordinal_columns, ordinal_categories):
    # Numerical pipeline
    numerical_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler(with_mean=False))
    ])
    
    # Categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ('scaler', StandardScaler(with_mean=False))
    ])
    
    # Ordinal pipeline
    ordinal_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=ordinal_categories)),
        ('scaler', StandardScaler(with_mean=False))
    ])
    
    # Column transformer
    preprocessor = ColumnTransformer([
        ('numerical_pipeline', numerical_pipeline, numerical_columns),
        ('categorical_pipeline', categorical_pipeline, categorical_columns),
        ('ordinal_pipeline', ordinal_pipeline, ordinal_columns)
    ],)
    
    return preprocessor

def transform_data(preprocessor, df_train, df_test, numerical_columns, categorical_columns, ordinal_columns, ordinal_categories):
    # Fit the preprocessor on the train data and transform both train and test data
    df_train_transformed = preprocessor.fit_transform(df_train)
    df_test_transformed = preprocessor.transform(df_test)

    # Get feature names for the OneHotEncoder
    # 'categorical_pipeline__onehot' gives access to the OneHotEncoder within the pipeline
    ohe_columns = list(preprocessor.named_transformers_['categorical_pipeline']['onehot'].get_feature_names_out(categorical_columns))

    # Combine all column names
    all_columns = numerical_columns + ohe_columns + ordinal_columns

    # Convert the NumPy arrays back to DataFrames with appropriate column names
    df_train_transformed = pd.DataFrame(df_train_transformed, columns=all_columns)
    df_test_transformed = pd.DataFrame(df_test_transformed, columns=all_columns)

    return df_train_transformed, df_test_transformed


def main():
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/processed/train_processed.csv')
        test_data = pd.read_csv('./data/processed/test_processed.csv')

        Road_traffic_density=['Low','Medium','High','Jam']
        Weather_conditions=['Sunny','Cloudy','Windy','Fog','Sandstorms','Stormy']

        categorical_columns=['Type_of_order','Type_of_vehicle','Festival','City','Delivery_city']
        ordinal_columns = ['Road_traffic_density', 'Weather_conditions']
        ordinal_categories = [Road_traffic_density, Weather_conditions]
        numerical_columns=['Delivery_person_Age','Delivery_person_Ratings','Vehicle_condition','multiple_deliveries',
                                'TimeOrder_Hour','distance']

        # separting the target column
        target_col_train = train_data['Time_taken (min)']
        target_col_test = test_data['Time_taken (min)']

        train_data.drop(['Time_taken (min)'], axis=1, inplace=True)
        train_data.drop(['Time_taken (min)'], axis=1, inplace=True)

        preprocessor = create_preprocessor(numerical_columns, categorical_columns, ordinal_columns, ordinal_categories)

        train_data_transformed, test_data_transformed = transform_data(preprocessor, train_data, test_data, numerical_columns,categorical_columns,
        ordinal_columns, ordinal_categories)

        # concating the target column 
        train_data_transformed = pd.concat([train_data_transformed, target_col_train], axis=1)
        test_data_transformed = pd.concat([test_data_transformed, target_col_test], axis=1)

        # Store the data inside data/interim
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_data_transformed.to_csv(os.path.join(data_path, "train_interim_transformed.csv"), index=False)
        test_data_transformed.to_csv(os.path.join(data_path, "test_interim_transformed.csv"), index=False)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()