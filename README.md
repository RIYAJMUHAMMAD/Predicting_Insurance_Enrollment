# Predicting_Insurance_Enrollment

## Overview

This notebook focuses on a classification task, likely predicting an enrollment status based on the provided data. The analysis includes data preprocessing, exploratory data analysis, model training using XGBoost with hyperparameter optimization, and model evaluation.


## Key Observations

* **Data Preprocessing:** Categorical features were successfully converted to numerical indices. No missing values were detected in the dataset. The target variable, Salary, exhibits a normal distribution.
* **Exploratory Data Analysis:** The target variable, Enrolled status, shows a class imbalance with an approximate ratio of 1:2.
* **Model Training:** An XGBoost classifier was trained due to time constraints. Hyperparameter optimization was performed using Optuna with 100 trials. The data was split into training and testing sets with a 60:40 ratio, employing stratified sampling to address the class imbalance in the test set.
* **Evaluation:** The model's performance was evaluated on both the training and testing sets using F1-score, precision, and recall, which are appropriate metrics for imbalanced datasets.
* **Future Work:** Given more time, exploring alternative models like Decision Trees or Logistic Regression for better explainability is recommended.


## Runing the api

1. `pip install requirements.txt`
2. `python app/app.py`

##Testing curl


curl -X POST -H "Content-Type: application/json" -d '{
    "age": 40,
    "gender": "Male",
    "marital_status": "Married",
    "salary": 5000,
    "employment_type": "Part-time",
    "region": "West",
    "has_dependents": "Yes",
    "tenure_years": 1.2
}' http://127.0.0.1:5000/predict
