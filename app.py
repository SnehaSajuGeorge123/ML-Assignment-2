
# Assignment 2: Machine Learning Classification Model Comparison

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="ML Classification Model Comparison",
    page_icon="📊",
    layout="wide"
)

st.title("Machine Learning Classification Model Comparison")

st.write(
    "This application evaluates five trained classification "
    "models using the default test dataset."
)

# --------------------------------------------------
# 2. Model file locations
# --------------------------------------------------

model_files = {
    "Logistic Regression": "Logistic_Regression.joblib",
    "Decision Tree Classifier": "Decision_Tree_Classifier.joblib",
    "K-Nearest Neighbor Classifier":
        "K_Nearest_Neighbor_Classifier.joblib",
    "Naive Bayes Classifier": "Naive_Bayes_Classifier.joblib",
    "Support Vector Machine": "Support_Vector_Machine.joblib"
}


@st.cache_resource
def load_model(model_name):
    filename = model_files[model_name]

    # Check the model folder first, then the root folder
    model_path = os.path.join("model", filename)

    if not os.path.exists(model_path):
        model_path = filename

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {filename}"
        )

    return joblib.load(model_path)


# --------------------------------------------------
# 3. Load the default test dataset
# --------------------------------------------------

st.subheader("Test Dataset")

try:
    test_data = pd.read_csv("test_data.csv")

    st.success("Default test dataset loaded successfully!")

    st.write("Dataset shape:", test_data.shape)

    st.dataframe(test_data.head(), use_container_width=True)

except Exception as error:
    st.error(f"Unable to load test_data.csv: {error}")
    st.stop()


# --------------------------------------------------
# 4. Separate input features and target
# --------------------------------------------------

if "target" not in test_data.columns:
    st.error("The test dataset must contain a 'target' column.")
    st.stop()

X_test = test_data.drop(columns=["target"])
y_test = test_data["target"]

if X_test.isnull().sum().sum() > 0:
    st.error("The test dataset contains missing feature values.")
    st.stop()

if y_test.isnull().sum() > 0:
    st.error("The target column contains missing values.")
    st.stop()


# --------------------------------------------------
# 5. Select a classification model
# --------------------------------------------------

st.subheader("Select a Classification Model")

selected_model = st.selectbox(
    "Choose one of the five trained models:",
    list(model_files.keys())
)


# --------------------------------------------------
# 6. Load model and generate predictions
# --------------------------------------------------

try:
    model = load_model(selected_model)

    expected_features = list(model.feature_names_in_)

    if set(X_test.columns) != set(expected_features):
        st.error(
            "The uploaded/default dataset features do not match "
            "the features expected by the selected model."
        )
        st.stop()

    # Arrange columns in the exact training order
    X_test = X_test[expected_features]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

except Exception as error:
    st.error(f"Model evaluation failed: {error}")
    st.stop()


# --------------------------------------------------
# 7. Calculate evaluation metrics
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

auc = roc_auc_score(y_test, y_prob)

precision = precision_score(
    y_test, y_pred, pos_label=1, zero_division=0
)

recall = recall_score(
    y_test, y_pred, pos_label=1, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, pos_label=1, zero_division=0
)

mcc = matthews_corrcoef(y_test, y_pred)

metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "AUC Score",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC"
    ],
    "Score": [
        accuracy,
        auc,
        precision,
        recall,
        f1,
        mcc
    ]
})


# --------------------------------------------------
# 8. Display model performance
# --------------------------------------------------

st.subheader("Model Performance")

st.write("Selected model:", selected_model)

st.dataframe(
    metrics_df.style.format({"Score": "{:.4f}"}),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 9. Display confusion matrix
# --------------------------------------------------

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Malignant", "Benign"],
    yticklabels=["Malignant", "Benign"],
    ax=ax
)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
ax.set_title(f"Confusion Matrix - {selected_model}")

st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# 10. Display classification report
# --------------------------------------------------

st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Malignant", "Benign"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# --------------------------------------------------
# 11. Evaluation information
# --------------------------------------------------

st.info(
    "The application uses the saved trained models and the "
    "default test dataset. No model retraining is performed. "
    "Precision, Recall and F1 Score use Benign (class 1) "
    "as the positive class."
)

st.success("Model evaluation completed successfully!")
