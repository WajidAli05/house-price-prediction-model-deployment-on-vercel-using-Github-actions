import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

# Load the dataset
file_path = "Housing.csv"  # Replace with your actual dataset path
housing_data = pd.read_csv(file_path)

# Define the categorical columns that need to be encoded
categorical_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]

# Use OneHotEncoder to encode categorical variables
onehot_encoder = OneHotEncoder(drop="first", sparse_output=False)
encoded_categorical_data = onehot_encoder.fit_transform(
    housing_data[categorical_columns]
)

# Create a dataframe with the encoded categorical variables
encoded_categorical_df = pd.DataFrame(
    encoded_categorical_data,
    columns=onehot_encoder.get_feature_names_out(categorical_columns),
)

# Drop the original categorical columns from the dataset and concatenate the encoded columns
housing_data_numeric = pd.concat(
    [housing_data.drop(columns=categorical_columns), encoded_categorical_df], axis=1
)

# Feature scaling (Standardizing numerical features)
scaler = StandardScaler()
# List of numerical columns that need scaling
numerical_columns = ["area", "bedrooms", "bathrooms", "stories", "parking"]
housing_data_numeric[numerical_columns] = scaler.fit_transform(
    housing_data_numeric[numerical_columns]
)

# Split the data into features (X) and target (y)
X = housing_data_numeric.drop(columns=["price"])
y = housing_data_numeric["price"]

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the Root Mean Squared Error (RMSE) for model evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse}")

# Save the trained model to a file (pkl format)
joblib.dump(model, "house_price_model.pkl")
print("Model saved as house_price_model.pkl")

# Save the scaler and encoder for future use in predictions
joblib.dump(scaler, "scaler.pkl")
joblib.dump(onehot_encoder, "encoder.pkl")
print("Scaler and Encoder saved for future use.")
