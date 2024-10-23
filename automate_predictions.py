import pandas as pd
import requests
from dbconnection import get_db_engine
from datetime import datetime


def get_new_data():
    engine = get_db_engine()
    query = "SELECT * FROM [dbo].[studentpattern] WHERE schoolcode = 498"
    df_new = pd.read_sql(query, engine)
    return df_new


def make_predictions(df_new):
    # Local Function URL
    function_url = 'http://localhost:7071/api/PredictScaleScore'
    predictions = []

    for index, row in df_new.iterrows():
        input_data = {
            "FiscalYear": row['FiscalYear'],
            "Gender": row['Gender'],
            "Grade": row['Grade'],
            "Ethnicity": row['Ethnicity'],
            "Assessment": row['Assessment'],
            "Subject": row['Subject'],
            "SE": row['SE'],
            "ELL": row['ELL'],
            "EconomicallyDisadvantaged": row['EconomicallyDisadvantaged'],
            "Migrant": row['Migrant'],
            "Homeless": row['Homeless'],
            "EnglishProficiency": row['EnglishProficiency'],
            "Achievement": row['Achievement'],
            "Achievement Level": row['Achievement Level']
        }

        response = requests.post(function_url, json={"data": input_data})
        prediction = response.json().get('prediction', None)  # Handle missing predictions
        predictions.append(prediction)

    df_new['Scale Score'] = predictions
    return df_new


def write_predictions(df_predictions):
    engine = get_db_engine()
    # Generate a unique table name or file name based on the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_name = f'StudentPattern_Predictions_{timestamp}'

    # Save predictions to a new table or file
    df_predictions.to_sql(table_name, con=engine, schema='stage', if_exists='replace', index=False)

    print(f"Predictions saved to table: {table_name}")


if __name__ == "__main__":
    df_new = get_new_data()
    if not df_new.empty:
        df_predictions = make_predictions(df_new)
        write_predictions(df_predictions)
        print("Predictions written to the database.")
    else:
        print("No new data to predict.")
