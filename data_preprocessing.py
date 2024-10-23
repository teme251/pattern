import pandas as pd
import numpy as np
from dbconnection import get_db_engine
from sklearn.preprocessing import LabelEncoder

# Load Data
def load_data():
    try:
        engine = get_db_engine()
        query = """
        SELECT ScaleScore, Gender, Grade, FiscalYear, Assessment, Subject, Migrant, Homeless, EnglishProficiency
        FROM dbo.studentpattern_school
        """
        df = pd.read_sql(query, engine)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

# Optimize data types
def optimize_types(df):
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')  # Downcast integers
        df[col] = pd.to_numeric(df[col], downcast='float')  # Downcast floats
    return df

# Handle Missing Values
def handle_missing_values(df):
    print("Handling missing values...")

    # Show columns with missing data
    missing_data = df.isna().sum()
    print(f"Columns with missing data:\n{missing_data[missing_data > 0]}")

    # Check if important columns exist
    important_columns = ['ScaleScore', 'Gender', 'Grade']
    existing_columns = [col for col in important_columns if col in df.columns]

    if len(existing_columns) > 0:
        print(f"Found the following important columns: {existing_columns}")
        df.dropna(subset=existing_columns, inplace=True)
    else:
        print("None of the important columns ('ScaleScore', 'Gender', 'Grade') found.")

    # Impute missing values for 'ScaleScore' if the column exists
    if 'ScaleScore' in df.columns:
        df['ScaleScore'] = pd.to_numeric(df['ScaleScore'], errors='coerce')  # Convert to numeric
        print(f"Imputing missing 'ScaleScore' with mean: {df['ScaleScore'].mean()}")
        df['ScaleScore'] = df['ScaleScore'].fillna(df['ScaleScore'].mean())  # Fill NaNs with the mean

    # Impute missing values for 'Gender' (fill with the most common gender or "Unknown")
    if 'Gender' in df.columns:
        most_frequent_gender = df['Gender'].mode()[0] if not df['Gender'].mode().empty else 'Unknown'
        print(f"Imputing missing 'Gender' with: {most_frequent_gender}")
        df['Gender'] = df['Gender'].fillna(most_frequent_gender)

    return df

# Encode Categorical Variables
def encode_categorical(df):
    categorical_columns = ['FiscalYear', 'Assessment', 'Subject', 'Migrant', 'Homeless', 'EnglishProficiency', 'Gender']
    existing_categorical_columns = [col for col in categorical_columns if col in df.columns]

    # Apply Label Encoding to these columns
    for col in existing_categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Convert to string first to handle any non-numeric issues

    return df

# Drop High Cardinality Columns
def drop_high_cardinality(df):
    # Drop columns with high cardinality
    high_cardinality_cols = ['School', 'SystemID', 'SchoolCode', 'SchoolKey', 'StudentKey']
    df.drop(columns=high_cardinality_cols, errors='ignore', inplace=True)
    return df

# Convert all columns to numeric, drop remaining NaN values, and print issues
def ensure_numeric(df):
    print("Converting all data to numeric...")

    # Convert to numeric and print any conversion issues
    df_numeric = df.apply(pd.to_numeric, errors='coerce')

    # Find problematic rows (rows with NaNs after conversion)
    problematic_rows = df_numeric[df_numeric.isna().any(axis=1)]
    if not problematic_rows.empty:
        print("Problematic rows after conversion to numeric:")
        print(problematic_rows)

    df_numeric.dropna(inplace=True)  # Drop rows with NaN values
    print(f"Final data shape after dropping rows with NaN values: {df_numeric.shape}")

    return df_numeric

# Main Preprocessing Function
def preprocess_data(df):
    if df.empty:
        print("No data to preprocess.")
        return df

    print(f"Initial data shape: {df.shape}")

    # Handle missing values
    df = handle_missing_values(df)

    # Optimize types for memory usage
    df = optimize_types(df)

    # Encode categorical variables
    df = encode_categorical(df)

    # Drop columns with high cardinality
    df = drop_high_cardinality(df)

    # Ensure all data is numeric
    df = ensure_numeric(df)

    print(f"Final data shape after preprocessing: {df.shape}")
    return df

# Run the preprocessing pipeline
if __name__ == "__main__":
    df = load_data()
    if not df.empty:
        print(f"Columns in the DataFrame: {df.columns}")  # Inspect column names
        df_clean = preprocess_data(df)
        if not df_clean.empty:
            # Save preprocessed data in a compressed format for later use
            df_clean.to_pickle('preprocessed_data.pkl', compression='gzip')
            print("Preprocessed data saved successfully.")
        else:
            print("Preprocessing failed. No data to save.")
    else:
        print("Failed to load data. Exiting.")
