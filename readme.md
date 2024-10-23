Explanation of Files:

dbconnection.py: Contains code to connect to your Azure SQL Database.
data_preprocessing.py: Script for loading and preprocessing data.
model_training.py: Script for training the machine learning model.
model_deployment/: Directory for Azure Function deployment.
__init__.py: Main function code for the Azure Function.
function.json: Configuration file for the Azure Function.
host.json: Host configuration for the Azure Function.
student_performance_model.pkl: Saved trained model.
model_features.pkl: Saved feature list used in the model.
automate_predictions.py: Script to fetch new data, make predictions, and write results back to the database.
requirements.txt: List of Python packages required for the project.
README.md: Documentation for your project.

# Student Pattern Project

## Project Structure

- `dbconnection.py`: Database connection.
- `data_preprocessing.py`: Data loading and preprocessing.
- `model_training.py`: Model training script.
- `model_deployment/`: Azure Function deployment files.
- `automate_predictions.py`: Script to automate predictions.
- `requirements.txt`: Python dependencies.

## Setup Instructions

1. **Install Dependencies**

```bash
pip install -r requirements.txt
