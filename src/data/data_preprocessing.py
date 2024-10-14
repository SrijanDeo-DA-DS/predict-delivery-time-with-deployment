import pandas as pd
import numpy as np
import math
import os

# Calculate distance using lat/long
def distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    R = 6371.0 # Earth's radius in km
    dist = R * c
    
    return dist

# Calculate Delivery city and Ratings
def cal_ratings(df, col1, col2):

    df['Delivery_city']=df[col1].str.split('RES',expand=True)[0]
    rating_map = round(df.groupby(col1)[col2].mean(),1).to_dict()
    df[col2]=df[col2].fillna(df[col1].map(rating_map))

    return df


# Extract HH/MM/YYYY from order date
def preprocess_date(df, column):
    df[column] = pd.to_datetime(df[column], format='%d-%m-%Y')

    df['year']= df[column].dt.year
    df['month']= df[column].dt.month
    df['day']= df[column].dt.day

    return df

# Preprocess Time ordered
def process_time_ordered(df, time_column):
    # Drop rows with NaN values in the time_column
    df.dropna(subset=[time_column], inplace=True)
    
    # Replace periods with colons in the time_column
    df[time_column] = df[time_column].str.replace('.', ':')
    
    # Define a function to extract the time in HH:MM format
    def extract_time(x):
        try:
            return x.split(':')[0] + ':' + x.split(':')[1][:2]
        except IndexError:
            return '00:00'

    # Apply the extract_time function
    df[time_column] = df[time_column].apply(extract_time)

    # Convert the time_column to datetime format, now using format '%H:%M'
    df[time_column] = pd.to_datetime(df[time_column], format='%H:%M', errors='coerce')
    
    # Ensure there are no conversion issues
    df.dropna(subset=[time_column], inplace=True)
    
    # Extract hour and handle cases where '0' should be replaced with '24'
    df['TimeOrder_Hour'] = df[time_column].dt.hour
    df['TimeOrder_Hour'] = df['TimeOrder_Hour'].replace(0, 24).astype(int)
    
    # Extract minutes from the time column
    df['TimeOrder_min'] = df[time_column].dt.minute
    
    return df

# Preprocess Time order picked
def process_time_order_picked(df, time_column):
    # Drop rows with NaN values in the time_column
    df.dropna(subset=[time_column], inplace=True)
    
    # Replace periods with colons in the time_column
    df[time_column] = df[time_column].str.replace('.', ':')
    
    # Define a function to extract the time in HH:MM format
    def extract_time(x):
        try:
            return x.split(':')[0] + ':' + x.split(':')[1][:2]
        except IndexError:
            return '00:00'

    # Apply the extract_time function
    df[time_column] = df[time_column].apply(extract_time)

    # Convert the time_column to datetime format, now using format '%H:%M'
    df[time_column] = pd.to_datetime(df[time_column], format='%H:%M', errors='coerce')
    
    # Ensure there are no conversion issues
    df.dropna(subset=[time_column], inplace=True)
    
    # Extract hour and handle cases where '0' should be replaced with '24'
    df['Time_Order_picked_Hour'] = df[time_column].dt.hour
    df['Time_Order_picked_Hour'] = df['Time_Order_picked_Hour'].replace(0, 24).astype(int)
    
    # Extract minutes from the time column
    df['Time_Order_picked_min'] = df[time_column].dt.minute

    return df


def preprocess_data(df):

    df['distance'] = df.apply(lambda row: distance(row['Restaurant_latitude'], row['Restaurant_longitude'], row['Delivery_location_latitude'], row['Delivery_location_longitude']), axis=1)

    df = cal_ratings(df, 'Delivery_person_ID', 'Delivery_person_Ratings')
    df = preprocess_date(df, 'Order_Date')
    df = process_time_ordered(df, 'Time_Orderd')

    df = process_time_order_picked(df, 'Time_Order_picked')

    df.drop(['Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude',
        'year','month','day','TimeOrder_min','Time_Order_picked_Hour','Time_Order_picked_min',
        'Time_Orderd','Time_Order_picked','Delivery_person_ID'],axis=1,inplace=True)
    
    return df

def main():
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')

        # Transform the data
        train_processed_data = preprocess_data(train_data)
        test_processed_data = preprocess_data(test_data)

        # Store the data inside data/processed
        data_path = os.path.join("./data", "processed")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()