import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

#For each dataset, we will load the data, clean it, merge it with the weather and holidays data, merge it with the city zones data, and remove outliers using the Isolation Forest model.
datasets = ['../Datasets/Small_dataset.parquet','../Datasets/Big_dataset.parquet']
for dataset in datasets:
    if os.path.exists(dataset):
        # Load the parquet dataset
        df = pd.read_parquet(dataset)

        # Drop rows with trip_distance <= 0
        df = df.drop(df[(df['trip_distance'] <= 0)].index)

        # Drop rows with fare_amount <= 0
        df = df[df['fare_amount'] > 0]
        #df = df[df['fare_amount'] <= 100]

        # Drop all the not needed columns
        df = df.drop(['total_amount'], axis=1)
        df = df.drop(['extra'], axis=1)
        df = df.drop(['mta_tax'], axis=1)
        df = df.drop(['tip_amount'], axis=1)
        df = df.drop(['tolls_amount'], axis=1)
        df = df.drop(['improvement_surcharge'], axis=1)
        df = df.drop(['congestion_surcharge'], axis=1)
        df = df.drop(['store_and_fwd_flag'], axis=1)
        df = df.drop(['payment_type'], axis=1)
        

        # Convert 'tpep_pickup_datetime' and 'tpep_dropoff_datetime' to datetime
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

        # Extract date and hour from 'tpep_pickup_datetime'
        df['pickup_date'] = df['tpep_pickup_datetime'].dt.date
        df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour

        # Calculate time spent in the taxi in minutes
        df['time_in_taxi'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

        # Drop unneeded columns
        df = df.drop(['tpep_pickup_datetime', 'tpep_dropoff_datetime'], axis=1)
        print("\nDropped 'tpep_pickup_datetime' and 'tpep_dropoff_datetime' columns:")
        
        # Drop rows with time_in_taxi <= 0 (negative values are not possible)
        df = df[df['time_in_taxi'] > 0]
    else:
        print("Dataset not found")

    # Merging the weather and holidays data
    weather = pd.read_csv('../Datasets/weather.csv')
    holidays = pd.read_csv('../Datasets/USHoliday.csv')

    # Drop columns not needed
    weather=weather.drop(columns=['tmax','tmin','departure','HDD','CDD'])

    # Convert 'date' to datetime to match the 'pickup_date' column
    weather['date'] = pd.to_datetime(weather['date']) 

    # Maintain only holidays in 2019
    holidays['Date'] = pd.to_datetime(holidays['Date'])
    holidays=holidays[holidays['Date'].dt.year==2019]

    #set precipitation to 0 if NaN and integer, new_snow, snow_depth
    weather['precipitation'] = weather['precipitation'].replace(to_replace="T", value=0)
    weather['new_snow'] = weather['new_snow'].replace(to_replace="T", value=0)
    weather['snow_depth'] = weather['snow_depth'].replace(to_replace="T", value=0)

    #set to float
    weather['precipitation'] = weather['precipitation'].astype(float)
    weather['new_snow'] = weather['new_snow'].astype(float)
    weather['snow_depth'] = weather['snow_depth'].astype(float)

    # Ensure the pickup_date column is in datetime64[ns] format
    df['pickup_date'] = pd.to_datetime(df['pickup_date'])

    # Merge the datasets on 'pickup_date' and 'date'
    new_df = pd.merge(df, weather, how='left', left_on='pickup_date', right_on='date')

    new_df = new_df.drop(['date'], axis=1)

    #Add day_time column: 1 if week day, 2 if weekend, 3 if holiday
    new_df['holiday'] = new_df['pickup_date'].isin(holidays['Date']).astype(int)
    new_df['day_of_week'] = new_df['pickup_date'].dt.dayofweek
    new_df['day_type'] = np.where(new_df['day_of_week'] < 5, 1, 2)
    new_df.loc[new_df['holiday'] == 1, 'day_type'] = 3
    new_df = new_df.drop(['pickup_date'], axis=1)
    new_df = new_df.drop(['day_of_week'], axis=1)
    new_df = new_df.drop(['holiday'], axis=1)
    new_df = new_df.dropna()

    # Merging the city zones data
    zones = pd.read_csv('../Datasets/taxi_zone_lookup.csv')
    zones = zones.drop(['Borough'], axis=1)
    zones = zones.drop(['Zone'], axis=1)

    zones = zones[zones['service_zone'] != 'N/A']

    # Replace 'EWR' with 'Airports' in the 'service_zone' column
    zones['service_zone'] = zones['service_zone'].replace('EWR', 'Airports')

    # Merge taxi_zone_lookup.csv with the new dataset on 'pulocationid' and 'dolocationid'
    pulocation = new_df.merge(zones[['LocationID', 'service_zone']], left_on='pulocationid', right_on='LocationID', how='left')
    dolocation = pulocation.merge(zones[['LocationID', 'service_zone']], left_on='dolocationid', right_on='LocationID', how='left', suffixes=('_pulocation', '_dolocation'))

    # Remove rows where 'zone_type' is None (rows that don't meet any of the conditions)
    new_df = dolocation
    new_df = new_df.drop(['pulocationid'], axis=1)
    new_df = new_df.drop(['dolocationid'], axis=1)
    new_df = new_df.drop(['LocationID_pulocation'], axis=1)
    new_df = new_df.drop(['LocationID_dolocation'], axis=1)
    
    # Replace 'service_zone' with numerical values
    new_df[['service_zone_pulocation', 'service_zone_dolocation']] = new_df[['service_zone_pulocation', 'service_zone_dolocation']].replace({'Airports': 1, 'Boro Zone': 2, 'Yellow Zone': 3})
    new_df = new_df.dropna()

    # Create and fit the Isolation Forest model
    # In order to remove outliers, we will use the Isolation Forest model from scikit-learn.
    iso_forest = IsolationForest(contamination=0.05, random_state=42)  # Adjust 'contamination' as needed
    outliers_pred = iso_forest.fit_predict(new_df)

    # Keep only inliers (label 1) and remove outliers (label -1)
    new_df_clean = new_df[outliers_pred == 1]

    # Check the result
    print("Original DataFrame shape:", new_df.shape)
    print("Cleaned DataFrame shape:", new_df_clean.shape)
    print(new_df_clean.head(1))
    dataset_path= dataset[:-8] + 'Preprocessed1.parquet'
    new_df_clean.to_parquet(dataset_path)