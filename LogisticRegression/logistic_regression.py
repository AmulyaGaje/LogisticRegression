import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config("Telco Customer Churn", layout="centered")

# ---------------- LOAD CSS ----------------
BASE_DIR = os.path.dirname(__file__)
CSS_PATH = os.path.join(BASE_DIR, "style.css")

def load_css(path):
    if os.path.exists(path):
        with open(path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css(CSS_PATH)

# ---------------- TITLE ----------------
st.markdown("""
<div class="card">
    <h1>TELCO CUSTOMER CHURN PREDICTION</h1>
    <p>Predict the <b>probability of customer churn</b> using Logistic Regression</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    csv_path = os.path.join(BASE_DIR, "Telco-Customer-Churn.csv")
    return pd.read_csv(csv_path)

df = load_data()

# ---------------- PREPROCESSING ----------------
df.drop("customerID", axis=1, inplace=True)

yes_no_cols = df.select_dtypes(include="object").columns
df[yes_no_cols] = df[yes_no_cols].replace({"Yes": 1, "No": 0})

df = pd.get_dummies(df, drop_first=True)

# ---------------- FEATURES & TARGET ----------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- MODEL ----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------- PREDICTIONS ----------------
y_pred = model.predict(X_test)

# ---------------- TASK 7: CONFUSION MATRIX ----------------
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Confusion Matrix Analysis")

c1, c2 = st.columns(2)
c1.metric("Correct Churn (TP)", tp)
c2.metric("Correct Non-Churn (TN)", tn)

c3, c4 = st.columns(2)
c3.metric("Missed Churn (FN)", fn)
c4.metric("Wrongly Flagged Loyal (FP)", fp)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TASK 8: BUSINESS INTERPRETATION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Business Interpretation")
st.write(f"- Wrongly flagged loyal customers as churn: **{fp}**")
st.write(f"- Missed customers who actually churned: **{fn}**")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PERFORMANCE METRICS ----------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

with st.container(border=True):
    st.subheader("Model Performance Metrics")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{accuracy:.2f}")
    c2.metric("Precision", f"{precision:.2f}")
    c3, c4 = st.columns(2)
    c3.metric("Recall", f"{recall:.2f}")
    c4.metric("F1 Score", f"{f1:.2f}")

# ---------------- PREDICTION SECTION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Customer Churn")

monthly_charges = st.slider(
    "Monthly Charges",
    float(df["MonthlyCharges"].min()),
    float(df["MonthlyCharges"].max()),
    70.0
)

tenure = st.slider(
    "Tenure (months)",
    int(df["tenure"].min()),
    int(df["tenure"].max()),
    12
)

# Create input row using mean values
input_data = X.mean().to_frame().T
input_data["MonthlyCharges"] = monthly_charges
input_data["tenure"] = tenure

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

result = "Churn" if prediction == 1 else "No Churn"

st.markdown(
    f'<div class="prediction-box">Prediction: <b>{result}</b></div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)

