import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

datasets = ['Datasets/Small_dataset.parquet','Datasets/Big_dataset.parquet']
for dataset in datasets:
    if os.path.exists(dataset):
        # Load the CSV using dask for parallel processing
        df = pd.read_parquet(dataset)

        df['total_amount'] = df['total_amount'] - df['tip_amount']

        # Drop rows where 'trip_distance' is 0 and 'fare_amount' is <= 0
        df = df.drop(df[(df['trip_distance'] <= 0)].index)
        # df = df.drop(['total_amount'], axis=1)
        df = df.drop(['extra'], axis=1)
        df = df.drop(['mta_tax'], axis=1)
        df = df.drop(['tip_amount'], axis=1)
        df = df.drop(['tolls_amount'], axis=1)
        df = df.drop(['improvement_surcharge'], axis=1)
        df = df.drop(['congestion_surcharge'], axis=1)
        df = df.drop(['store_and_fwd_flag'], axis=1)
        df = df.drop(['payment_type'], axis=1)
        # df = df.drop(['fare_amount'], axis=1)
        df = df[df['fare_amount'] > 0]
        df = df.drop(['fare_amount'], axis=1)
        #df = df[df['fare_amount'] <= 100]


        # Convert 'tpep_pickup_datetime' and 'tpep_dropoff_datetime' to datetime
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
        print("Datetime conversion completed:")
        print(df[['tpep_pickup_datetime', 'tpep_dropoff_datetime']].head())

        # Extract date and hour from 'tpep_pickup_datetime'
        df['pickup_date'] = df['tpep_pickup_datetime'].dt.date
        df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
        print("\nExtracted 'pickup_date' and 'pickup_hour':")
        print(df[['pickup_date', 'pickup_hour']].head())

        # Calculate time spent in the taxi in minutes
        df['time_in_taxi'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
        print("\nCalculated 'time_in_taxi':")
        print(df[['time_in_taxi']].head())

        # Drop unneeded columns
        df = df.drop(['tpep_pickup_datetime', 'tpep_dropoff_datetime'], axis=1)
        print("\nDropped 'tpep_pickup_datetime' and 'tpep_dropoff_datetime' columns:")
        print(df.head())
        df = df[df['time_in_taxi'] > 0]

        #print the data distribution over base on pickup_date as a line plot only in 2019
        plt.figure(figsize=(12, 6))
        plt.plot(df['pickup_date'].value_counts().sort_index())
        plt.title('Distribution of Pickup Date')
        plt.xlim(pd.Timestamp('2019-01-01'), pd.Timestamp('2019-12-31'))
        plt.savefig('pickup_date_distribution.png')
        plt.show()
        #print the data distribution over base on pickup_hour
        plt.figure(figsize=(12, 6))
        plt.plot(df['pickup_hour'].value_counts().sort_index())
        plt.title('Distribution of Pickup Hour')
        plt.savefig('pickup_hour_distribution.png')
        plt.show()
        # Merging the weather and holidays data
        weather = pd.read_csv('Datasets/weather.csv')

        weather=weather.drop(columns=['tmax','tmin','departure','HDD','CDD'])
        weather['date'] = pd.to_datetime(weather['date'])

        holidays = pd.read_csv('Datasets/USHoliday.csv')

        #maintain only if holiday is in 2019
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

        new_df = pd.merge(df, weather, how='left', left_on='pickup_date', right_on='date')

        new_df = new_df.drop(['date'], axis=1)
        #add column 1 if week day, 2 if weekend, 3 if holiday
        new_df['holiday'] = new_df['pickup_date'].isin(holidays['Date']).astype(int)
        new_df['day_of_week'] = new_df['pickup_date'].dt.dayofweek
        new_df['day_type'] = np.where(new_df['day_of_week'] < 5, 1, 2)
        new_df.loc[new_df['holiday'] == 1, 'day_type'] = 3
        new_df = new_df.drop(['pickup_date'], axis=1)
        new_df = new_df.drop(['day_of_week'], axis=1)
        new_df = new_df.drop(['holiday'], axis=1)
        new_df = new_df.dropna()

        # Merging the city zones data
        zones = pd.read_csv('Datasets/taxi_zone_lookup.csv')
        zones = zones.drop(['Borough'], axis=1)
        zones = zones.drop(['Zone'], axis=1)

        zones = zones[zones['service_zone'] != 'N/A']

        # Replace 'EWR' with 'Airports' in the 'service_zone' column
        zones['service_zone'] = zones['service_zone'].replace('EWR', 'Airports')

        # Merge taxi_zone_lookup.csv with the new dataset on 'pulocationid' and 'dolocationid'
        pulocation = new_df.merge(zones[['LocationID', 'service_zone']], left_on='pulocationid', right_on='LocationID', how='left')
        dolocation = pulocation.merge(zones[['LocationID', 'service_zone']], left_on='dolocationid', right_on='LocationID', how='left', suffixes=('_pulocation', '_dolocation'))

        # Create a new column 'zone_type' based on the conditions
        def get_zone_type(row):
            service_zone_pulocation = row['service_zone_pulocation']
            service_zone_dolocation = row['service_zone_dolocation']

            if service_zone_pulocation == 'Airports' or service_zone_dolocation == 'Airports':
                return 1
            elif 'Boro Zone' in [service_zone_pulocation, service_zone_dolocation]:
                return 2
            elif 'Yellow Zone' in [service_zone_pulocation, service_zone_dolocation]:
                return 3
            else:
                return None

        # Apply the zone_type function to the merged dataframe
        #dolocation['zone_type'] = dolocation.apply(get_zone_type, axis=1)

        # Remove rows where 'zone_type' is None (rows that don't meet any of the conditions)
        new_df = dolocation
        print(new_df.head(1))
        new_df = new_df.drop(['pulocationid'], axis=1)
        new_df = new_df.drop(['dolocationid'], axis=1)
        new_df = new_df.drop(['LocationID_pulocation'], axis=1)
        new_df = new_df.drop(['LocationID_dolocation'], axis=1)
        #new_df = new_df.drop(['service_zone_pulocation'], axis=1)
        #new_df = new_df.drop(['service_zone_dolocation'], axis=1)
        new_df[['service_zone_pulocation', 'service_zone_dolocation']] = new_df[['service_zone_pulocation', 'service_zone_dolocation']].replace({'Airports': 1, 'Boro Zone': 2, 'Yellow Zone': 3})
        new_df = new_df.dropna()
        print(new_df.head(1))
        print(new_df.shape)    

        # Create and fit the Isolation Forest model
        iso_forest = IsolationForest(contamination=0.05, random_state=42)  # Adjust 'contamination' as needed
        outliers_pred = iso_forest.fit_predict(new_df)

        # Keep only inliers (label 1) and remove outliers (label -1)
        new_df_clean = new_df[outliers_pred == 1]

        # Check the result
        print("Original DataFrame shape:", new_df.shape)
        print("Cleaned DataFrame shape:", new_df_clean.shape)
        #delete when total_amount is > 350
        new_df_clean = new_df_clean[new_df_clean['total_amount'] <= 350]
        print("Cleaned DataFrame shape:", new_df_clean.shape)
        dataset_path= dataset[:-8] + 'Preprocessed.parquet'
        new_df_clean.to_parquet(dataset_path)
        print(new_df.head())
    else:
        print("Dataset not found")
    