import streamlit as st
import pandas as pd
from xgboost import XGBClassifier

# Load model safely (JSON format avoids corruption issues)
model = XGBClassifier()
model.load_model("1st_best_model.json")

st.title("Fraud Detection System")

# --- Input fields ---
amount = st.number_input("Transaction Amount", min_value=0.0, step=10.0)
transaction_hour = st.number_input("Transaction Hour (0-23)", min_value=0, max_value=23, step=1)
foreign_transaction = st.selectbox("Foreign Transaction", [0, 1])
location_mismatch = st.selectbox("Location Mismatch", [0, 1])
device_trust_score = st.number_input("Device Trust Score", min_value=0.0, step=1.0)
velocity_last_24h = st.number_input("Transactions in Last 24h", min_value=0, step=1)
cardholder_age = st.number_input("Cardholder Age", min_value=18, max_value=100, step=1)

merchant_electronics = st.checkbox("Merchant Category: Electronics")
merchant_food = st.checkbox("Merchant Category: Food")
merchant_grocery = st.checkbox("Merchant Category: Grocery")
merchant_travel = st.checkbox("Merchant Category: Travel")

if st.button("Predict"):
    sample_dict = {
        "amount": amount,
        "transaction_hour": transaction_hour,
        "foreign_transaction": foreign_transaction,
        "location_mismatch": location_mismatch,
        "device_trust_score": device_trust_score,
        "velocity_last_24h": velocity_last_24h,
        "cardholder_age": cardholder_age,
        "merchant_category_Electronics": int(merchant_electronics),
        "merchant_category_Food": int(merchant_food),
        "merchant_category_Grocery": int(merchant_grocery),
        "merchant_category_Travel": int(merchant_travel),
    }

    sample_df = pd.DataFrame([sample_dict])

    # --- Backend threshold (hidden from frontend) ---
    threshold = 0.3   # chosen after tuning
    prob = model.predict_proba(sample_df)[0][1]
    pred = int(prob >= threshold)

    # --- Output ---
    st.write("### Prediction:", "Fraud" if pred == 1 else "Not Fraud")
    st.write("Risk Estimate:", round(prob * 100, 2), "%")
