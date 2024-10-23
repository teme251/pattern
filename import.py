import pandas as pd

# Read the CSV file
df_new_data = pd.read_csv('path_to_your_csv_file.csv')
from dbconnection import get_db_engine

engine = get_db_engine()

# Write data to SQL table
df_new_data.to_sql(
    name='YourNewTable',
    con=engine,
    schema='schema',
    if_exists='append',  # Use 'replace' if you want to overwrite the table
    index=False,         # Do not write DataFrame index as a column
    chunksize=1000       # Number of rows to write at a time
)
