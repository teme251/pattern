# model_deployment/PredictScaleScore/__init__.py

import logging
import azure.functions as func
import json
import pandas as pd
import joblib
import os

# Load the model and features during cold start
MODEL_FILE = os.path.join(os.path.dirname(__file__), '..', 'student_performance_model.pkl')
FEATURES_FILE = os.path.join(os.path.dirname(__file__), '..', 'model_features.pkl')

model = joblib.load(MODEL_FILE)
model_features = joblib.load(FEATURES_FILE)


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Received a request for predicting Scale Score.')

    try:
        # Parse the JSON request body
        req_body = req.get_json()
        input_data = req_body['data']

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Perform one-hot encoding on input data
        input_encoded = pd.get_dummies(input_df)

        # Align the input data with the training data features
        input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

        # Make prediction
        prediction = model.predict(input_encoded)

        # Return the prediction as JSON
        return func.HttpResponse(
            json.dumps({'prediction': prediction[0]}),
            status_code=200,
            mimetype='application/json'
        )

    except Exception as e:
        logging.error(str(e))
        return func.HttpResponse(
            json.dumps({'error': str(e)}),
            status_code=400,
            mimetype='application/json'
        )
