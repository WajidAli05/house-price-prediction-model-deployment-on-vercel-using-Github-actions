import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the dataset
file_path = 'Housing.csv'  # Replace with your actual dataset path
housing_data = pd.read_csv(file_path)

# Define the categorical columns that need to be encoded
categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

# Use OneHotEncoder to encode categorical variables
onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_categorical_data = onehot_encoder.fit_transform(housing_data[categorical_columns])

# Create a dataframe with the encoded categorical variables
encoded_categorical_df = pd.DataFrame(encoded_categorical_data, columns=onehot_encoder.get_feature_names_out(categorical_columns))

# Drop the original categorical columns from the dataset and concatenate the encoded columns
housing_data_numeric = pd.concat([housing_data.drop(columns=categorical_columns), encoded_categorical_df], axis=1)

# Feature scaling (Standardizing numerical features)
scaler = StandardScaler()
# List of numerical columns that need scaling
numerical_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
housing_data_numeric[numerical_columns] = scaler.fit_transform(housing_data_numeric[numerical_columns])

# Now the dataset `housing_data_numeric` is preprocessed and ready for model training
