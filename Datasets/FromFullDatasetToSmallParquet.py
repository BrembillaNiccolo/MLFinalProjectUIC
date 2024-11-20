import pandas as pd
import os

# Set the downloaded files from the Kaggle repository in parquet folder
# This code will read all the parquet files and create a new parquet file with 100,000 rows
parquet_paths = [
    '2019-01',
    '2019-02',
    '2019-03',
    '2019-04',
    '2019-05',
    '2019-06',
    '2019-07',
    '2019-08',
    '2019-09',
    '2019-10',
    '2019-11',
    '2019-12'
]

df = pd.DataFrame()

for path in parquet_paths:
    parquetPath = 'parquet_files/' + path + '.parquet'
    print(parquetPath)
    
    if os.path.exists(parquetPath):
        # Load data into a pandas DataFrame
        new_df = pd.read_parquet(parquetPath)
        # Get 100,000 rows randomly for each DataFrame
        new_df = new_df.sample(n=100000, random_state=42)
        # Concatenate data
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        print(f"Parquet file not found at {parquetPath}")

# Save the concatenated DataFrame to a new Parquet file
df.to_parquet('Small_dataset.parquet', index=False)
