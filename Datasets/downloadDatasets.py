import kagglehub
import pandas as pd
import os
import sqlite3

# Step 1: Download the latest version of the dataset
path = kagglehub.dataset_download("dhruvildave/new-york-city-taxi-trips-2019")
print("Path to dataset files:", path)

# Step 2: Define a function to convert SQLite tables to Parquet
def convert_sqlite_to_parquet(sqlite_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate over all SQLite files in the directory
    for filename in os.listdir(sqlite_directory):
        if filename.endswith('.sqlite'):
            sqlite_path = os.path.join(sqlite_directory, filename)
            print(f"Processing {sqlite_path}...")

            # Connect to the SQLite database
            conn = sqlite3.connect(sqlite_path)
            cursor = conn.cursor()

            # Get the list of tables in the database
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            # Iterate over each table in the SQLite database
            for table in tables:
                table_name = table[0]
                print(f"Extracting table {table_name}...")

                # Query the table into a pandas DataFrame
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                #remove .sqlite from filename
                filename = filename.replace('.sqlite','')
                # Define output path with .parquet extension
                parquet_filename = f"{filename}.parquet"
                parquet_path = os.path.join(output_directory, parquet_filename)

                # Write the DataFrame to a Parquet file
                df.to_parquet(parquet_path, index=False)
                print(f"Saved {parquet_path}")

            # Close the SQLite connection
            conn.close()

# Step 3: Convert all CSVs to Parquet files and save them in the current directory
output_path = os.path.join(os.getcwd(), 'parquet_files')  # Save in 'parquet_files' folder in the current directory
path=os.path.join(path, '2019')
convert_sqlite_to_parquet(path, output_path)
print("All CSV files have been converted to Parquet format.")
