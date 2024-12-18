## **ImmoEliza Real Estate Price Prediction Tool**

### **Description**

ImmoElizaPredict is a predictive tool designed to estimate real estate prices in Belgium based on various property characteristics.  
It uses a machine learning model such as Random Forest to provide accurate price predictions.  
The application is implemented using Streamlit for an interactive user interface  
and supports clean data processing, preprocessing and model prediction.

### **Folder and File Structure**

```plaintext
Immo_Deployment/  
    ├── asset/   
    │   ├── predict.png    
    │   ├── screenshot.png     
    ├── data/                               #    Contains the processed dataset for training and predictions  
    │   ├── immoweb_data_cleaned.csv
    │   ├── immoweb_data_processed.csv
    │   ├── zips.csv
    ├── models/                             # Includes the Random Forest model script (random_forest_model.py)  
    │                                       # and the serialized model (random_forest_model.joblib)  
    │   ├── random_forest_model.py  
    │   ├── random_forest_model.joblib  
    ├── predict/                            # Contains the prediction.py script used for loading the model and making predictions  
    │   ├── prediction.py  
    ├── preprocessing/                      # Contains the preprocessing.py script used for cleaning dataset  
    │   ├── preprocessing.py  
    ├── app.py                              # The main Streamlit application file for the interactive user interface  
    ├── README.md                           # Documentation for the project  
    ├── requirements.txt                    # Documentation of Python dependencies required to run the project  
```

## **Installation**

1. Clone the repository:

        git clone https://github.com/FomAndrii/immo_eliza_ML  
        cd ImmoEliza_ML  

2. Set up a virtual environment and install dependencies:

        python -m venv venv  
        source venv/bin/activate                # On Windows: venv\Scripts\activate  
        pip install -r requirements.txt  

3. Ensure the dataset immoweb_data_processed.csv is in the data directory and the Random Forest model random_forest_model.joblib is in the models directory.

### **Usage**

Run the Streamlit app:

            streamlit run app.py

Open the APP [here](https://immoelizaandriideployment.streamlit.app/).

Input property details, such as locality, type, number of rooms, living area, and kitchen status, to predict the price.

### **Visuals**

**![The main page](<asset/screenshot.png>)**

### **Contributors**

**Andrii Fomichov:** BeCode learner, code implementor and lead of data deployment.  
**Joen Cheramy:** Coaching and feedback.  
**Yassine Sekri:**  Coaching and motivation.  

### **Timeline**

**Day 1:** Data cleaning and preprocessing.  
**Day 2-3:** Feature engineering, correlation analysis, and dummy variable creation.  
**Day 4:** Deployment on Streamlit.io.  
**Day 5-6** Documentation, optimization, and project presentation.  

### **Personal Situation**

This project was developed as part of an AI Bootcamp, focusing on regression in machine learning such as Random Forest. It serves as a portfolio piece to demonstrate expertise in Python programming, data preprocessing, machine learning model development, and interactive app deployment. Working on ImmoEliza has been a rewarding journey that combined technical skills with problem-solving for real-world scenarios.
