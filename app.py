from flask import Flask, render_template, request
import numpy as np
import joblib

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
 # Prepare input data
        input_features = [area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]
        
        # Preprocess categorical and numerical features
        categorical_values = input_features[4:]
        encoded_categorical_values = encoder.transform([categorical_values])
        
        numerical_values = np.array(input_features[:4] + [parking]).reshape(1, -1)
        scaled_numerical_values = scaler.transform(numerical_values)
        
        # Combine both preprocessed categorical and numerical values
        preprocessed_input = np.hstack((scaled_numerical_values, encoded_categorical_values))
        
        # Predict the price
        predicted_price = model.predict(preprocessed_input)
        
        return render_template('index.html', result=predicted_price[0])

if __name__ == '__main__':
    app.run(debug=True)
