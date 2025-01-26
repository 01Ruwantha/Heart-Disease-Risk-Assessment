
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("heart_disease_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/', methods=['GET', 'POST'])
def predict():
    result = None
    probability = None

    if request.method == 'POST':
        # Get form data
        features = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        input_data = [float(request.form[feature]) for feature in features]

        # Prepare the input data for prediction
        new_data = np.array([input_data])

        # Scale the data
        new_data_scaled = scaler.transform(new_data)

        # Predict using the trained model
        prediction = model.predict(new_data_scaled)
        probability = model.predict_proba(new_data_scaled)[0][1]

        # Determine result
        result = "High Risk" if prediction[0] == 1 else "Low Risk"

        # Round the probability for display
        probability = round(probability, 2)

    return render_template("index.html", result=result, probability=probability)

if __name__ == '__main__':
    app.run(debug=True)
