import streamlit as st
import pandas as pd
import joblib

model = joblib.load("adaboost.joblib")
def get_prediction(data:pd.DataFrame):
    pred = model.predict(data)
    pred_proba = model.predict_proba(data)
    return pred, pred_proba

st.title("Bank Service Churn Predictor")
st.subheader("🏦 Predict whether a customer will churn or not")


# User Input
st.divider()
st.subheader("👫 Customer Profile")
surname = st.text_input("Surname")
gender = st.selectbox("Gender", ["Male", "Female"], index=None)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"], index=None)
age = st.slider("Age", min_value=18, max_value=100, step=1, value=25)

st.divider()
st.subheader("💸 Financial and Business")
active_member = st.selectbox("Active Member", ["yes", "no"], index=None)
credit_card = st.selectbox("Credit Card", ["yes", "no"], index=None)
num_of_products = st.slider("Number of Products", min_value=0, max_value=10, step=1, value=1)
tenure = st.slider("Tenure", min_value=0, max_value=100, step=1 ,value=1)
balance = st.number_input("Balance", step=1, value = None)
estimated_salary = st.number_input("Estimated Salary", step=1, value=None)
credit_score = st.number_input("Credit Score", step=1, value=None)



if credit_card == "yes":
    credit_card = 1
else:
    credit_card = 0

if active_member == "yes":
    active_member = 1
else:
    active_member = 0


data = pd.DataFrame({"Surname": [surname],
    "Gender": [gender],
    "Geography": [geography],
    "Age": [age],
    "IsActiveMember": [active_member],
    "HasCrCard": [credit_card],
    "NumOfProducts": [num_of_products],
    "Tenure": [tenure],
    "Balance": [balance],
    "EstimatedSalary": [estimated_salary],
    "CreditScore": [credit_score]})


st.divider()
st.subheader("📋 Customer Data")
st.dataframe(data, use_container_width=True, hide_index=True)

# Prediction
st.divider()
st.subheader("📊 Prediction")
button =st.button("Predict", use_container_width=True)

try:
    if button:
        pred, pred_proba = get_prediction(data)
        if pred == 1:
            st.success("Customer will churn")
            st.write(f"with {round(pred_proba[0][1] * 100)} % confidence")
        else:
            st.success("Customer will not churn")
            st.write(f"with {100 - round(pred_proba[0][1] * 100)} % confidence")
except:
    st.error("Please fill in all the fields")

