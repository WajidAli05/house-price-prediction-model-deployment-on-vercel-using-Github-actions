import pandas as pd
import numpy as np
import joblib

# Load the trained model, scaler, and encoder
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

# Load the dataset for testing (this can be a new dataset or a split test set)
file_path = 'Housing.csv' 
housing_data = pd.read_csv(file_path)

# Preprocess categorical columns (using the same steps as in training)
categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
encoded_categorical_data = encoder.transform(housing_data[categorical_columns])

# Create a dataframe with the encoded categorical variables
encoded_categorical_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_columns))

# Drop original categorical columns and concatenate the encoded columns
housing_data_numeric = pd.concat([housing_data.drop(columns=categorical_columns), encoded_categorical_df], axis=1)

# Scale the numerical columns (using the saved scaler)
numerical_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
housing_data_numeric[numerical_columns] = scaler.transform(housing_data_numeric[numerical_columns])

# Remove the target column if it exists (only if the file has price column, skip if testing on new data)
X_test = housing_data_numeric.drop(columns=['price'])

# Make predictions using the loaded model
predictions = model.predict(X_test)

# Output the predictions
print(f"Predicted house prices: {predictions}")
