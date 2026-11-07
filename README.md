
# Machine Learning Classification Model Comparison

## 1. Problem Statement

The objective of this project is to implement and compare five machine learning classification models using the Wisconsin Breast Cancer Diagnostic Dataset. The models are evaluated using six performance metrics.

## 2. Dataset Description

- Dataset: Wisconsin Breast Cancer Diagnostic Dataset
- Number of instances: 569
- Number of features: 30
- Target variable: target
- Classes: Malignant (0) and Benign (1)
- Missing values: 0

The dataset contains numerical measurements computed from digitized images of breast mass samples.

## 3. Machine Learning Models

The following five classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier (Gaussian)
5. Support Vector Machine

## 4. Evaluation Metrics

The models were evaluated using:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

Precision, recall and F1 score use Benign (class 1) as the positive class. AUC is calculated using the predicted probability of the Benign class.

## 5. Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree Classifier | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| K-Nearest Neighbor Classifier | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes Classifier | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

## 6. Model Performance Observations

- Logistic Regression achieved the highest AUC score of 0.9954 and an accuracy of 98.25%. It correctly classified 112 out of 114 test instances.
- Decision Tree Classifier achieved the lowest accuracy of 91.23% and incorrectly classified 10 test instances.
- K-Nearest Neighbor Classifier achieved 95.61% accuracy and 97.88% AUC, showing good overall performance.
- Naive Bayes Classifier achieved 93.86% accuracy and 98.78% AUC.
- Support Vector Machine achieved 98.25% accuracy and 99.50% AUC, matching Logistic Regression on the other five metrics.

## 7. Overall Winner

Logistic Regression is selected as the overall winner because it achieved the highest AUC score of 99.54%, while matching Support Vector Machine on accuracy, precision, recall, F1 score and MCC.

## 8. Streamlit Application

The Streamlit application allows users to:

- Upload a test CSV file.
- Select one of the five trained classification models.
- View the six evaluation metrics.
- View the confusion matrix.
- View the classification report.

The application loads the saved trained models and evaluates them on the uploaded test dataset without retraining.

## 9. Project Files

- `app.py` - Streamlit application.
- `requirements.txt` - Required Python libraries.
- `test_data.csv` - Test dataset.
- `model/` - Saved trained classification models.
