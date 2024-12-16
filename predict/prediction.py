# predict/prediction.py

import joblib
import pandas as pd

# Load the trained Random Forest model
model = joblib.load('models/random_forest_model.joblib')

def predict(input_data):
    """
    Predict the price of a real estate property using the trained Random Forest model.

    Args:
    - input_data (dict): A dictionary containing the features of the property.
        The keys should match the feature names used in the model.

    Returns:
    - float: The predicted price of the property.
    """
    # Convert the input data into a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make the prediction using the Random Forest model
    predicted_price = model.predict(input_df)
    
    return predicted_price[0]
