import tensorflow as tf
from tensorflow.keras.models import load_model 
import pickle 
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


# Load the trained model, scaler and encoders

model = load_model('model.h5')

with open('labelencoder.pkl', 'rb') as file:
    labelencoder = pickle.load(file)

with open("onehotencoder.pkl", "rb") as file:
    onehotencoder = pickle.load(file)
    
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


## streamlit app
st.title("Customer Churn Prediction")

## User input 
geography = st.selectbox("Geography", onehotencoder.categories_[0])

gender = st.selectbox("Gender", labelencoder.classes_)
age = st.slider("Age", 18, 100)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")

estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

## Preprocess user input
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [labelencoder.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

geo_encoded_df = onehotencoder.transform(input_data[["Geography"]])
geo_encoded_df.head()

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.drop("Geography",axis=1), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')