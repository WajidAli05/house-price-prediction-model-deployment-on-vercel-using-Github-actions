from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model, scaler, and encoder
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Retrieve form data
            area = int(request.form['area'])
            bedrooms = int(request.form['bedrooms'])
            bathrooms = int(request.form['bathrooms'])
            stories = int(request.form['stories'])
            mainroad = request.form['mainroad']
            guestroom = request.form['guestroom']
            basement = request.form['basement']
            hotwaterheating = request.form['hotwaterheating']
            airconditioning = request.form['airconditioning']
            parking = int(request.form['parking'])
            prefarea = request.form['prefarea']
            furnishingstatus = request.form['furnishingstatus']

            # Prepare input data for encoding and scaling
            categorical_values = [mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus]
            numerical_values = [area, bedrooms, bathrooms, stories, parking]

            # Encode categorical features using the pre-trained encoder
            categorical_df = pd.DataFrame([categorical_values], columns=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'])
            encoded_categorical_values = encoder.transform(categorical_df)

            # Scale numerical features using the pre-trained scaler
            numerical_values = np.array(numerical_values).reshape(1, -1)
            scaled_numerical_values = scaler.transform(numerical_values)

            # Combine both preprocessed categorical and numerical values
            preprocessed_input = np.hstack((scaled_numerical_values, encoded_categorical_values))

            # Predict the price using the trained model
            predicted_price = model.predict(preprocessed_input)

            # Render the template and pass the predicted price to it
            return render_template('index.html', result=f"${predicted_price[0]:,.2f}")

        except Exception:
            # Handle the exception by showing an error message
            return render_template('index.html', result="Error occurred during prediction.")


if __name__ == '__main__':
    app.run(debug=True)
