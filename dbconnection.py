"""Azure SQL connection helper for the student performance prototype.

Set AZURE_SQL_SERVER, AZURE_SQL_DATABASE, AZURE_SQL_USERNAME, and
AZURE_SQL_PASSWORD in your environment. Never commit credentials.
"""
import os
import urllib.parse

import pandas as pd
from sqlalchemy import create_engine, text


def get_db_engine():
    names = {
        "server": "AZURE_SQL_SERVER",
        "database": "AZURE_SQL_DATABASE",
        "username": "AZURE_SQL_USERNAME",
        "password": "AZURE_SQL_PASSWORD",
    }
    values = {key: os.environ.get(name) for key, name in names.items()}
    missing = [names[key] for key, value in values.items() if not value]
    if missing:
        raise RuntimeError("Missing required environment variables: " + ", ".join(missing))

    driver = os.environ.get("AZURE_SQL_DRIVER", "ODBC Driver 17 for SQL Server")
    params = urllib.parse.quote_plus(
        f"DRIVER={{{driver}}};"
        f"SERVER={values['server']};"
        f"DATABASE={values['database']};"
        f"UID={values['username']};"
        f"PWD={values['password']}"
    )
    return create_engine(f"mssql+pyodbc:///?odbc_connect={params}")


def test_connection():
    engine = get_db_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT count(1) FROM dbo.studentpattern_school"))
        print("Rows:", result.scalar_one())
        df = pd.read_sql("SELECT TOP 1 * FROM dbo.studentpattern_school", conn)
        print(df.head())


if __name__ == "__main__":
    test_connection()
