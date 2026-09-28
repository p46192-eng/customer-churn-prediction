import streamlit as st
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(
    page_title="Customer Churn Decision Support",
    page_icon="📊",
    layout="wide",
)

@st.cache_resource
def load_model():
    dataset = fetch_ucirepo(id=563)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    y.columns = ["Churn"]

    id_columns = [c for c in X.columns if str(c).strip().lower() in {"customer id", "customer_id", "id"}]
    if id_columns:
        X = X.drop(columns=id_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y["Churn"].astype(int),
        test_size=0.20,
        random_state=42,
        stratify=y["Churn"].astype(int),
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }

    return model, list(X.columns), metrics


st.title("📊 Customer Churn Prediction")
st.subheader("Managerial Decision-Support Tool")

st.write(
    "Estimate a customer's churn probability using historical customer "
    "behaviour. The result is intended to support prioritisation and "
    "managerial review—not replace managerial judgement."
)

st.info(
    "Dataset: Iranian Churn — UCI Machine Learning Repository. "
    "Customer behaviour variables are based on the first 9 months, while "
    "the churn outcome represents the customer's state at month 12."
)

try:
    model, feature_names, metrics = load_model()
except Exception as e:
    st.error(
        "The model could not be loaded from the UCI dataset. "
        "Please check the internet connection and package versions."
    )
    st.stop()

st.markdown("### Customer information")

# Defaults are deliberately neutral and editable. They are not presented as a
# recommended customer profile.
defaults = {
    "Call Failure": 5,
    "Complains": 0,
    "Subscription Length": 12,
    "Charge Amount": 2,
    "Seconds of Use": 3000,
    "Frequency of use": 50,
    "Frequency of SMS": 20,
    "Distinct Called Numbers": 20,
    "Age Group": 3,
    "Tariff Plan": 1,
    "Status": 1,
    "Age": 30,
    "Customer Value": 300.0,
}

inputs = {}
cols = st.columns(3)

for i, feature in enumerate(feature_names):
    default = defaults.get(feature, 0)
    label = feature

    if feature in {"Complains"}:
        value = cols[i % 3].selectbox(
            label,
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            index=int(default),
        )
    elif feature == "Tariff Plan":
        value = cols[i % 3].selectbox(
            label,
            options=[1, 2],
            format_func=lambda x: "Pay as you go" if x == 1 else "Contractual",
            index=0 if default == 1 else 1,
        )
    elif feature == "Status":
        value = cols[i % 3].selectbox(
            label,
            options=[1, 2],
            format_func=lambda x: "Active" if x == 1 else "Non-active",
            index=0 if default == 1 else 1,
        )
    elif feature == "Age Group":
        value = cols[i % 3].selectbox(
            label,
            options=[1, 2, 3, 4, 5],
            index=int(default) - 1,
            help="1 = younger age group; 5 = older age group.",
        )
    elif feature == "Charge Amount":
        value = cols[i % 3].number_input(
            label, min_value=0, max_value=9, value=int(default), step=1
        )
    elif feature == "Subscription Length":
        value = cols[i % 3].number_input(
            label, min_value=0, max_value=100, value=int(default), step=1
        )
    elif feature == "Age":
        value = cols[i % 3].number_input(
            label, min_value=0, max_value=100, value=int(default), step=1
        )
    else:
        step = 0.1 if feature == "Customer Value" else 1
        value = cols[i % 3].number_input(
            label,
            min_value=0.0 if feature == "Customer Value" else 0,
            value=float(default) if feature == "Customer Value" else int(default),
            step=step,
        )

    inputs[feature] = value

if st.button("Predict Churn Risk", type="primary", use_container_width=True):
    input_df = pd.DataFrame([inputs], columns=feature_names)
    probability = float(model.predict_proba(input_df)[0, 1])

    if probability >= 0.70:
        risk = "HIGH RISK"
        interpretation = (
            "The model estimates a relatively high probability of churn. "
            "Review the customer's engagement, complaints, usage and value "
            "before deciding whether a retention intervention is appropriate."
        )
    elif probability >= 0.40:
        risk = "MEDIUM RISK"
        interpretation = (
            "The model identifies a moderate churn probability. "
            "The customer may warrant additional review before prioritising "
            "a retention intervention."
        )
    else:
        risk = "LOW RISK"
        interpretation = (
            "The model estimates a relatively low probability of churn. "
            "This does not mean churn is impossible; continue normal "
            "customer monitoring and use business context."
        )

    st.markdown("---")
    st.markdown("### Prediction")

    c1, c2 = st.columns(2)
    c1.metric("Estimated churn probability", f"{probability:.1%}")
    c2.metric("Risk category", risk)

    st.progress(probability)
    st.write(interpretation)

    st.warning(
        "Managerial note: A prediction is a screening signal, not a certainty. "
        "Where the model conflicts with customer history or managerial "
        "experience, investigate further before taking action."
    )

with st.expander("Model performance"):
    metric_cols = st.columns(5)
    for col, (name, value) in zip(metric_cols, metrics.items()):
        col.metric(name, f"{value:.3f}")

with st.expander("How to interpret the tool"):
    st.markdown(
        """
        **What the tool does:** estimates the probability of the documented
        churn outcome from the customer's historical behaviour.

        **What it does not do:** guarantee that a customer will churn or
        prescribe a specific retention action.

        **Recommended managerial workflow:**
        1. Use the prediction to prioritise customers for review.
        2. Examine customer context and recent interactions.
        3. Consider the cost and feasibility of a retention intervention.
        4. Make the final decision using managerial judgement.

        Risk thresholds in this interface (70% and 40%) are communication
        thresholds for this academic prototype, not universal industry standards.
        """
    )

with st.expander("Dataset & methodology"):
    st.write(
        "The application uses the Iranian Churn dataset from the UCI Machine "
        "Learning Repository (Dataset 563). The dataset contains 3,150 "
        "customers and 13 predictor variables. UCI reports no missing values. "
        "Predictors represent the first nine months and the churn label "
        "represents the customer's state at month 12."
    )
    st.write(
        "Model: Logistic Regression with StandardScaler preprocessing and "
        "balanced class weights. Evaluation uses a stratified 80/20 train-test split."
    )
    st.markdown(
        "[UCI dataset documentation](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset)"
    )
