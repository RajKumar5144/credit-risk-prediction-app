import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.set_page_config(page_title="Credit Risk App", layout="centered")

st.title("💳 Credit Risk Prediction App")
st.write("Predict loan default risk using ML model")

# ---------------- INPUTS ----------------
st.subheader("📝 Enter Applicant Details")

age = st.slider("Age", 18, 75, 25)
income = st.number_input("Annual Income", 1000, 300000, 50000)
emp_length = st.slider("Employment Length (years)", 0, 40, 5)
loan_amount = st.number_input("Loan Amount", 500, 50000, 10000)
interest_rate = st.slider("Interest Rate (%)", 5.0, 25.0, 12.0)

home_ownership = st.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

loan_intent = st.selectbox(
    "Loan Purpose",
    ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
)

loan_grade = st.selectbox(
    "Loan Grade",
    ["A","B","C","D","E","F","G"]
)

# ---------------- FEATURE ENGINEERING ----------------
income_to_loan_ratio = income / loan_amount
stable_employment = 1 if emp_length >= 5 else 0
high_interest_flag = 1 if interest_rate > 15 else 0

# ---------------- CREATE INPUT DF ----------------
input_data = pd.DataFrame({
    'person_age':[age],
    'person_income':[income],
    'person_emp_length':[emp_length],
    'loan_amnt':[loan_amount],
    'loan_int_rate':[interest_rate],
    'income_to_loan_ratio':[income_to_loan_ratio],
    'stable_employment':[stable_employment],
    'high_interest_flag':[high_interest_flag],
    'person_home_ownership':[home_ownership],
    'loan_intent':[loan_intent],
    'loan_grade':[loan_grade]
})

# Encoding
input_encoded = pd.get_dummies(input_data)
input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Risk"):

    prob = model.predict_proba(input_encoded)[0][1]
    risk_score = int(prob * 1000)

    # Risk band & decision
    if risk_score < 300:
        risk_band = "Low Risk"
        decision = "Approve ✅"
    elif risk_score < 650:
        risk_band = "Medium Risk"
        decision = "Manual Review ⚠️"
    else:
        risk_band = "High Risk"
        decision = "Reject ❌"

    # ---------------- RESULTS ----------------
    st.subheader("📊 Prediction Result")

    # Risk category display
    if risk_band == "High Risk":
        st.error("❌ High Risk - Very likely to default")
    elif risk_band == "Medium Risk":
        st.warning("⚠️ Medium Risk - Needs review")
    else:
        st.success("✅ Low Risk - Safe applicant")

    st.write(f"**Probability of Default:** {prob:.2%}")
    st.write(f"**Risk Score:** {risk_score}")
    st.write(f"**Loan Decision:** {decision}")

    # ---------------- CHART 1: RISK INDICATOR ----------------
    st.subheader("🎯 Risk Score Indicator")

    st.progress(risk_score / 1000)
    st.metric("Risk Score", risk_score)
    st.metric("Default Probability", f"{prob:.2%}")

    # ---------------- CHART 2: PROBABILITY COMPARISON ----------------
    st.subheader("📊 Default vs Non-Default Probability")

    prob_df = pd.DataFrame({
        "Category": ["Non-Default", "Default"],
        "Probability": [1 - prob, prob]
    })

    st.bar_chart(prob_df.set_index("Category"))

    # ---------------- CHART 3: KEY INPUT FACTORS ----------------
    st.subheader("📌 Key Applicant Factors")

    factors = pd.DataFrame({
        "Feature": ["Income", "Loan Amount", "Interest Rate", "Employment Length"],
        "Value": [income, loan_amount, interest_rate, emp_length]
    })

    st.bar_chart(factors.set_index("Feature"))

    # ---------------- INTERPRETATION ----------------
    st.subheader("🧠 Model Interpretation")

    if prob > 0.7:
        st.write("🔴 High probability of default — risky applicant.")
    elif prob > 0.3:
        st.write("🟡 Moderate risk — further review recommended.")
    else:
        st.write("🟢 Low risk — applicant likely safe.")