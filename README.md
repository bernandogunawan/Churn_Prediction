# 🏦 Bank Churn Predictor (Streamlit App)

This Streamlit app predicts whether a **bank customer will churn (leave)** or **stay** based on their profile and account details.  
It’s designed for **bank staff, analysts, or data scientists** to quickly test predictions using manual inputs or batch data uploads.


Try it here:
https://churnprediction-bernando.streamlit.app/

## 🚀 Features

- **Step-based Navigation**
  - **Home** → Overview of the app and required data.
  - **Insert Data** → Add customer details manually or upload batch CSV/Excel.
  - **Review** → Inspect and clean the input dataset before prediction.
  - **Predict** → Run churn predictions and visualize results.

- **Input Methods**
  - Manual form entry for one customer at a time.
  - Batch upload using `.csv` or `.xlsx` files.

- **Prediction**
  - Uses a trained **AdaBoost model** (`adaboost.joblib`) to classify churn vs. stay.
  - Adjustable probability threshold to fine-tune classification.
  - Visual outputs (pie chart distribution of predictions).

- **Export**
  - Download results as a CSV file after prediction.

---

## 📊 Required Input Data

The model expects the following columns:

| Column            | Description |
|-------------------|-------------|
| **Surname**       | Customer’s last name |
| **Gender**        | Male / Female |
| **Geography**     | France, Spain, Germany |
| **Age**           | Age (18–100) |
| **CreditScore**   | Bank-assigned numeric score |
| **Tenure**        | Years with the bank (0–50) |
| **Balance**       | Current balance (USD) |
| **NumOfProducts** | Number of products (0–10) |
| **HasCrCard**     | yes / no |
| **IsActiveMember**| yes / no |
| **EstimatedSalary**| Approx. yearly salary |

