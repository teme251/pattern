from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = RandomForestRegressor(n_estimators=100, random_state=42)  # Example: Your trained model
scaler = StandardScaler()  # Example: Pre-trained scaler

# Load your pre-trained model from file if saved
# with open('model.pkl', 'rb') as f:
#    model = pickle.load(f)

@app.route('/model_training_app', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json()

    # Convert data into DataFrame or numpy array for prediction
    df = pd.DataFrame([data])

    # Preprocess the input data as per your model requirements (e.g., scaling)
    X_scaled = scaler.transform(df)

    # Perform prediction
    predictions = model.predict(X_scaled)

    # Send back the prediction
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
