# 💳 Credit Risk Prediction App

An end-to-end Machine Learning project to predict loan default risk using customer financial and demographic data. The model is deployed using Streamlit with an interactive dashboard for real-time predictions.

---

## 🚀 Features

- Predicts probability of loan default
- Generates a risk score (0–1000)
- Classifies applicants into:
  - Low Risk
  - Medium Risk
  - High Risk
- Provides loan decision:
  - Approve
  - Manual Review
  - Reject
- Interactive dashboard using Streamlit
- Visual risk interpretation (charts & indicators)

---

## 🧠 Machine Learning Model

- Model Used: LightGBM Classifier
- Handling Imbalance: SMOTE
- Evaluation Metrics:
  - ROC-AUC: ~0.94
  - KS Statistic: ~74%
  - F1 Score: ~0.83

---

## 📊 Key Features Used

- Person Income
- Loan Amount
- Interest Rate
- Employment Length
- Income to Loan Ratio (Engineered)
- Credit History Length
- Loan Grade
- Home Ownership

---

## ⚙️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- LightGBM
- Streamlit
- Matplotlib

---

## 📁 Project Structure
credit-risk-prediction-app/
│
├── app.py
├── model.pkl
├── columns.pkl
├── requirements.txt
├── credit_risk.ipynb
└── dataset



---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py


