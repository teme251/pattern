import sqlalchemy
import pyodbc
import pandas as pd
from sqlalchemy import create_engine, text
import urllib

def get_db_engine():
    # Database credentials
    server = 'teambitembank.database.windows.net'
    database = 'AI-DataSupport'
    username = 'teambadmin'
    password = 'Team@Admin'
    driver = '{ODBC Driver 17 for SQL Server}'

    # Build the connection string
    params = urllib.parse.quote_plus(
        f'DRIVER={driver};'
        f'SERVER={server};'
        f'DATABASE={database};'
        f'UID={username};'
        f'PWD={password}'
    )

    engine = create_engine(f'mssql+pyodbc:///?odbc_connect={params}')
    return engine


# Function to test the connection by fetching one row
def test_connection():
    engine = get_db_engine()

    # Execute query using the text() method
    with engine.connect() as conn:
        result = conn.execute(text("SELECT count(1) FROM dbo.studentpattern_school"))
        for row in result:
            print(row)
            query = "SELECT count(1) FROM [dbo].[studentpattern_school]"
            df = pd.read_sql(query, engine)

            print(df.head())
            print(df.info())
            print(df.describe())


# Call the test function
test_connection()

