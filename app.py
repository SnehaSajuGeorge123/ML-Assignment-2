
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

st.set_page_config(
    page_title="ML Classification Model Comparison",
    page_icon="📊",
    layout="wide"
)

st.title("Machine Learning Classification Model Comparison")
st.write(
    "Upload a test dataset, select one of the five trained models, "
    "and view its classification performance."
)

# Model names and saved file paths
model_files = {
    "Logistic Regression": "model/Logistic_Regression.joblib",
    "Decision Tree Classifier": "model/Decision_Tree_Classifier.joblib",
    "K-Nearest Neighbor Classifier": "model/K_Nearest_Neighbor_Classifier.joblib",
    "Naive Bayes Classifier": "model/Naive_Bayes_Classifier.joblib",
    "Support Vector Machine": "model/Support_Vector_Machine.joblib"
}

# Upload test dataset
uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.write("Dataset shape:", test_data.shape)
    st.dataframe(test_data.head())

    if "target" not in test_data.columns:
        st.error("The uploaded CSV must contain a 'target' column.")

    else:
        X_test = test_data.drop(columns=["target"])
        y_test = test_data["target"]

        selected_model_name = st.selectbox(
            "Select a classification model",
            list(model_files.keys())
        )

        model_path = model_files[selected_model_name]

        if not os.path.exists(model_path):
            st.error("Saved model file not found: " + model_path)

        else:
            model = joblib.load(model_path)

            try:
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

                # Calculate evaluation metrics
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

                st.subheader("Model Performance")
                st.write("Selected model:", selected_model_name)

                metrics_data = pd.DataFrame({
                    "Metric": [
                        "Accuracy", "AUC Score", "Precision",
                        "Recall", "F1 Score", "MCC"
                    ],
                    "Score": [
                        accuracy, auc, precision, recall, f1, mcc
                    ]
                })

                st.dataframe(
                    metrics_data.style.format({"Score": "{:.4f}"}),
                    use_container_width=True
                )

                # Confusion matrix
                st.subheader("Confusion Matrix")

                cm = confusion_matrix(y_test, y_pred)

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
                ax.set_title("Confusion Matrix - " + selected_model_name)

                st.pyplot(fig)
                plt.close(fig)

                # Classification report
                st.subheader("Classification Report")

                report = classification_report(
                    y_test,
                    y_pred,
                    target_names=["Malignant", "Benign"],
                    output_dict=True,
                    zero_division=0
                )

                report_df = pd.DataFrame(report).transpose()
                st.dataframe(
                    report_df.round(4),
                    use_container_width=True
                )

            except Exception as error:
                st.error(
                    "Unable to evaluate the model. "
                    "Please check that the uploaded CSV has the correct "
                    "feature columns and target values."
                )
                st.write("Details:", error)

else:
    st.info("Please upload test_data.csv to begin the evaluation.")
