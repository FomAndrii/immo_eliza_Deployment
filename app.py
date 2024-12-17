import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from predict.prediction import predict

# Load a logo Immoweb predict
st.image("./asset/predict.png")

# Load the dataset for UI options
data_path = "data/immoweb_data_processed.csv"
data = pd.read_csv(data_path)

# Streamlit UI elements
st.header("Immoweb, Belgium’s top real estate site for over 20 years, has created Immoweb (:blue[ImmoEliza]) Predict, "
        "a Machine Learning Model, to forecast price of your favourite property.", divider=True)
st.subheader(":blue[Thousands of customers already trust our forecasting.]:sunglasses:")

# Input fields for property details
locality = st.selectbox("Locality", data["Locality"].unique())
property_type = st.selectbox("Property Type", ["APARTMENT", "HOUSE"])
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

# Display a feedback widget with stars, and show the selected sentiment:
st.markdown(''':blue-background[Please, give me your feedback about our forecasting and read my Regular Column: AI in Law]''')
sentiment_mapping = ["one", "two", "three", "four", "five"]
selected = st.feedback("stars")
if selected is not None:
    st.markdown(f"You selected {sentiment_mapping[selected]} star(s).")