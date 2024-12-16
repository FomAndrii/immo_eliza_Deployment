import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from predict.prediction import predict

# Load the dataset for UI options
data_path = "data/immoweb_data_processed.csv"
data = pd.read_csv(data_path)

# Streamlit UI elements
st.title("ImmoEliza Real Estate Price Prediction Tool")
st.write("Welcome! Use this tool to estimate the price of a property based on various features.")

# Input fields for property details
locality = st.selectbox("Locality", data["Locality"].unique())
property_type = st.selectbox("Property Type", ["HOUSE", "APARTMENT"])
num_rooms = st.number_input("Number of Rooms", min_value=1, max_value=7, step=1)
living_area = st.number_input("Living Area (in sqm)", min_value=10, max_value=1000, step=10)
kitchen = st.selectbox("Fully Equipped Kitchen", ["Yes", "No"])

# Encode 'Type_of_Property' using Label Encoding
le = LabelEncoder()
property_type_encoded = le.fit_transform(["HOUSE", "APARTMENT"])  # Fit on the categories
property_type_dict = dict(zip(["HOUSE", "APARTMENT"], property_type_encoded))  # Create a mapping

# Create a dictionary of inputs for prediction
input_data = {
    "Locality": locality,
    "Type_of_Property": property_type_dict[property_type],  # Use encoded value
    "Number_of_Rooms": num_rooms,
    "Living_Area": living_area,
    "Fully_Equipped_Kitchen": 1 if kitchen == "Yes" else 0
}

# Button to trigger prediction
if st.button("Predict Price"):
    try:
        # Make prediction
        predicted_price = predict(input_data)
        st.success(f"The predicted price for the property is: €{predicted_price:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
