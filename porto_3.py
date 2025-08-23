import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ---------- Page Setup ----------
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide"
)

# ---------- Load Model ----------
def load_model(path="adaboost.joblib"):
    return joblib.load(path)

model = load_model()

# ---------- Helper Functions ----------
def get_prediction(df: pd.DataFrame):
    pred = model.predict(df)
    proba = model.predict_proba(df)
    return pred, proba

def yes_no_to_int(int):
    if int == "yes":
        return 1
    else:
        return 0

# ---------- Session State ----------
if "input_data" not in st.session_state:
    st.session_state.input_data = pd.DataFrame()
if "current_step" not in st.session_state:
    st.session_state.current_step = "Home"

# ---------- Sidebar as Vertical Steps ----------
st.sidebar.title("📌 Steps")
steps = ["Home", "Insert Data", "Review", "Predict"]
st.session_state.current_step = st.sidebar.radio("Go to step:", steps, index=steps.index(st.session_state.current_step))

# ---------- STEP: Home ----------
if st.session_state.current_step == "Home":
    st.title("🏦 Bank Service Churn Predictor")
    st.subheader("Predict whether a customer will churn or stay")
    st.markdown("""
    This app helps bank staff or analysts **predict customer churn**.  

    **Steps:**  
    1. Insert customer data (manual or batch upload)  
    2. Review the data  
    3. Predict churn and visualize results  

    **Required Data Columns:**  
    - `Surname` → Customer’s last name.  
    - `Gender` *(Male/Female)* → Customer’s gender.  
    - `Geography` *(France, Spain, Germany)* → Country where the customer is located.  
    - `Age` *(18–100)* → Customer’s age.  
    - `CreditScore` *(numeric)* → Bank-assigned score reflecting creditworthiness.  
    - `Tenure` *(0–10 years typical)* → How many years the customer has been with the bank.  
    - `Balance` *(numeric, in USD)* → Customer’s current account balance.  
    - `NumOfProducts` *(integer)* → Number of bank products the customer uses (e.g., savings, loan, credit card).  
    - `HasCrCard` *(yes/no)* → Whether the customer has a credit card.  
    - `IsActiveMember` *(yes/no)* → Whether the customer is considered active (frequent usage of bank services).  
    - `EstimatedSalary` *(numeric)* → Approximate yearly salary of the customer.  

    👉 For **batch upload**, make sure your CSV/Excel file contains these columns with the correct names and formats.  

    Use the sidebar to navigate between steps.
    """
)

# ---------- STEP: Insert Data ----------
elif st.session_state.current_step == "Insert Data":
    st.header("1️⃣ Insert Customer Data")
    insert_type = st.radio("Insert Type:", ["Manual Input", "Batch Upload"])

    if insert_type == "Manual Input":
        left, right = st.columns(2)
        with left:
            surname = st.text_input("Surname (optional)", placeholder="e.g. Smith",)
            gender = st.selectbox("Gender", ["Male", "Female"], index=None)
            geography = st.selectbox("Geography", ["France", "Spain", "Germany"], index=None)
            age = st.slider("Age", 18, 100, 25)
        with right:
            active_member = st.selectbox("Active Member", ["yes","no"], index=None)
            credit_card = st.selectbox("Credit Card", ["yes","no"], index=None)
            num_of_products = st.slider("Number of Products", 0, 10, 1)
            tenure = st.slider("Tenure (years)", 0, 50, 1)
            balance = st.number_input("Balance", step=1.0, value=None, placeholder="e.g. 1000.0")
            estimated_salary = st.number_input("Estimated Salary", step=1.0, value=None, placeholder="e.g. 5000.0")
            credit_score = st.number_input("Credit Score", step=1.0, value=None, placeholder="e.g. 600.0")

        if st.button("Add to Dataset"):
            # Validation checks
            errors = []
            if not gender:
                errors.append("Gender is required.")
            if not geography:
                errors.append("Geography is required.")
            if age is None:
                errors.append("Age is required.")
            if credit_score is None or credit_score == 0:
                errors.append("Credit Score is required and must be > 0.")
            if tenure is None:
                errors.append("Tenure is required.")
            if balance is None:
                errors.append("Balance is required.")
            if estimated_salary is None:
                errors.append("Estimated Salary is required.")
            if num_of_products is None:
                errors.append("Number of Products is required.")
            if active_member not in ["yes", "no"]:
                errors.append("Active Member must be 'yes' or 'no'.")
            if credit_card not in ["yes", "no"]:
                errors.append("Credit Card must be 'yes' or 'no'.")

            if errors:
                for e in errors:
                    st.error(e)
            else:
                # If all checks pass, add row
                row = pd.DataFrame({
                    "Surname": [surname if surname else "N/A"],
                    "Gender": [gender],
                    "Geography": [geography],
                    "Age": [age],
                    "IsActiveMember": [yes_no_to_int(active_member)],
                    "HasCrCard": [yes_no_to_int(credit_card)],
                    "NumOfProducts": [num_of_products],
                    "Tenure": [tenure],
                    "Balance": [balance],
                    "EstimatedSalary": [estimated_salary],
                    "CreditScore": [credit_score]
                })
                st.session_state.input_data = pd.concat([st.session_state.input_data, row], ignore_index=True)
                st.success("✅ Customer added successfully!")


    else:  # Batch upload
        uploaded_file = st.file_uploader("Upload CSV/Excel for batch prediction", type=['csv','xlsx'])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".csv"):
                    batch_data = pd.read_csv(uploaded_file)
                else:
                    batch_data = pd.read_excel(uploaded_file)
                st.session_state.input_data = batch_data
                st.success("Batch data loaded!")
            except Exception as e:
                st.error(f"Error reading file: {e}")

# ---------- STEP: Review ----------
elif st.session_state.current_step == "Review":
    st.header("2️⃣ Review Data")
    if st.session_state.input_data.empty:
        st.info("No data to review. Go to 'Insert Data' step to add customers.")
    else:
        st.dataframe(st.session_state.input_data, use_container_width=True, hide_index=True)
        if st.button("Clear Dataset"):
            st.session_state.input_data = pd.DataFrame()
            st.success("Dataset cleared!")

# ---------- STEP: Predict ----------
elif st.session_state.current_step == "Predict":
    st.header("3️⃣ Predict & Visualize")
    if st.session_state.input_data.empty:
        st.info("No data to predict. Please insert data first.")
    else:
        threshold = st.slider("⚙️ Set Churn Probability Threshold", 0.0, 1.0, 0.5, 0.01)
        if st.button("🔮 Predict"):
            try:
                data = st.session_state.input_data.copy()

                pred, pred_proba = get_prediction(data)
                churn_probs = pred_proba[:,1]
                stay_probs = pred_proba[:,0]
                decisions = ["CHURN" if p>=threshold else "STAY" for p in churn_probs]

                data["Prediction"] = decisions
                data["Churn_Probability"] = churn_probs

                st.subheader("📋 Predictions")

                st.subheader("🍩 Prediction Distribution")
                if len(data) == 1:
                    st.success(f"Prediction: {decisions[0]}")
                    counts = [stay_probs[0], churn_probs[0]]
                else:
                    st.dataframe(data, use_container_width=True, hide_index=True)
                    counts = [decisions.count("STAY"), decisions.count("CHURN")]
                    
                labels = ["Stay", "Churn"]
                colors = ['#4CAF50', '#F44336']

                fig, ax = plt.subplots()
                ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90,
                       colors=colors, wedgeprops=dict(width=0.4))
                ax.axis('equal')
                st.pyplot(fig)

                st.download_button(
                    "📥 Download Results as CSV",
                    data.to_csv(index=False),
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
