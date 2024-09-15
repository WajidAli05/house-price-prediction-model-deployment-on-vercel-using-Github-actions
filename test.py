import unittest
import joblib
import numpy as np
import pandas as pd
import os


class TestHousePriceModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure the model and preprocessors are available
        cls.model_path = "house_price_model.pkl"
        cls.scaler_path = "scaler.pkl"
        cls.encoder_path = "encoder.pkl"

    def test_preprocessing(self):
        """Test the preprocessing (OneHotEncoding and Scaling) of the dataset."""
        # Sample input for preprocessing
        sample_input = [
            5000, 4, 3, 2, "yes", "no", "no", "no", "yes", 2, "yes", "furnished"
        ]

        # Load the saved scaler and encoder
        scaler = joblib.load(self.scaler_path)
        encoder = joblib.load(self.encoder_path)

        # Ensure the correct number of features are preprocessed
        categorical_columns = [
            "mainroad", "guestroom", "basement", "hotwaterheating",
            "airconditioning", "prefarea", "furnishingstatus"
        ]
        numerical_columns = ["area", "bedrooms", "bathrooms", "stories", "parking"]

        # Split the categorical and numerical values from sample_input
        categorical_values = [
            sample_input[4], sample_input[5], sample_input[6],
            sample_input[7], sample_input[8], sample_input[10],
            sample_input[11]
        ]
        numerical_values = [
            sample_input[0], sample_input[1], sample_input[2],
            sample_input[3], sample_input[9]
        ]

        # Use DataFrame with column names for encoding and scaling
        categorical_df = pd.DataFrame([categorical_values], columns=categorical_columns)
        numerical_df = pd.DataFrame([numerical_values], columns=numerical_columns)

        # Test encoding and scaling
        encoded_categorical_values = encoder.transform(categorical_df)
        scaled_numerical_values = scaler.transform(numerical_df)

        # Ensure the transformation result is of the correct size
        self.assertEqual(
            encoded_categorical_values.shape[1],
            len(encoder.get_feature_names_out(categorical_columns)),
            "Unexpected number of categorical encoded features"
        )
        self.assertEqual(
            scaled_numerical_values.shape[1],
            len(numerical_columns),
            "Unexpected number of numerical features after scaling"
        )

    def test_model_training_and_saving(self):
        """Test if the model can be trained and saved properly."""
        # Ensure the model is saved correctly after training
        self.assertTrue(os.path.exists(self.model_path), "Model file was not saved")
        self.assertTrue(os.path.exists(self.scaler_path), "Scaler file was not saved")
        self.assertTrue(os.path.exists(self.encoder_path), "Encoder file was not saved")

    def test_model_loading_and_prediction(self):
        """Test if the saved model can be loaded and used for prediction."""
        # Load the model, scaler, and encoder
        model = joblib.load(self.model_path)
        scaler = joblib.load(self.scaler_path)
        encoder = joblib.load(self.encoder_path)

        # Define categorical and numerical columns
        categorical_columns = [
            "mainroad", "guestroom", "basement", "hotwaterheating",
            "airconditioning", "prefarea", "furnishingstatus"
        ]
        numerical_columns = ["area", "bedrooms", "bathrooms", "stories", "parking"]

        # Sample input for prediction (same structure as the original data)
        sample_input = [
            5000, 4, 3, 2, "yes", "no", "no", "no", "yes", 2, "yes", "furnished"
        ]

        # Preprocess input (categorical encoding and scaling)
        categorical_values = [
            sample_input[4], sample_input[5], sample_input[6],
            sample_input[7], sample_input[8], sample_input[10],
            sample_input[11]
        ]
        numerical_values = [
            sample_input[0], sample_input[1], sample_input[2],
            sample_input[3], sample_input[9]
        ]

        # Use DataFrame with column names for encoding and scaling
        categorical_df = pd.DataFrame([categorical_values], columns=categorical_columns)
        numerical_df = pd.DataFrame([numerical_values], columns=numerical_columns)

        # Apply transformations
        encoded_categorical_values = encoder.transform(categorical_df)
        scaled_numerical_values = scaler.transform(numerical_df)

        # Combine preprocessed data for model input
        model_input = np.hstack((scaled_numerical_values, encoded_categorical_values))

        # Make a prediction
        prediction = model.predict(model_input)

        # Check that the prediction is a non-negative number (since house prices can't be negative)
        self.assertGreaterEqual(prediction[0], 0, "Predicted price is negative")

    def test_model_persistence(self):
        """Test if the model and preprocessors are saved as .pkl files."""
        # Check if the model, scaler, and encoder are saved
        self.assertTrue(
            os.path.exists(self.model_path), "Model was not saved correctly"
        )
        self.assertTrue(
            os.path.exists(self.scaler_path), "Scaler was not saved correctly"
        )
        self.assertTrue(
            os.path.exists(self.encoder_path), "Encoder was not saved correctly"
        )


if __name__ == "__main__":
    unittest.main()
