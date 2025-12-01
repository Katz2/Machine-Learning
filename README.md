# Machine-Learning
Patient Readmission Prediction — Machine Learning Case Study
Predicting 30-day hospital readmissions using Python, Pandas, SKLearn, and Machine Learning
Project Overview
This project analyses a hospital dataset to predict whether a patient will be readmitted within 30 days of discharge.
Unnecessary readmissions are costly for healthcare providers, and early prediction can improve both patient outcomes and operational efficiency.
The goal is to build and evaluate a supervised machine learning model using Python’s data science ecosystem.

Dataset Summary
The dataset includes:
Age
Gender
Admission Type
Length of Stay
Number of Diagnoses
Blood Pressure
Blood Sugar Levels
Previous Admissions
Readmission (Target: Yes/No)

1. Data Preprocessing
✔ Handled missing values (mean/median imputation and category completion)
✔ Normalised continuous variables (StandardScaler / MinMaxScaler)
✔ Encoded categorical features (One-Hot and Label Encoding)
✔ Removed or capped outliers in vital signs
✔ Ensured dataset was ready for ML modeling

2. Exploratory Data Analysis (EDA)
Performed EDA using Matplotlib, Seaborn, and Databricks visualizations:
Distribution of key variables (Age, Length of Stay, Diagnoses)
Boxplots for detecting outliers
Scatterplots showing relationships between demographics and readmission
Correlation matrix for numeric variables
Bar charts for admission type and gender breakdown
Feature interactions
Early patterns in readmission probability

3. Feature Selection
Techniques used:
Correlation analysis
Chi-square tests (for categorical variables)
Feature importance from tree-based models
Domain-driven justification
Most influential predictors identified:
Length of Stay
Number of Diagnoses
Blood Sugar Levels
Previous Admissions
Age
Admission Type

4. Model Development
Used Scikit-Learn to build:
Logistic Regression
Decision Tree Classifier
Random Forest Classifier
Also implemented:
Train/Test Split
Cross-validation
Hyperparameter tuning using GridSearchCV
Pipeline automation

5. Model Evaluation
Evaluated models using:
Accuracy
Precision
Recall
F1-Score
ROC-AUC Curve
Random Forest achieved the best performance with a strong ROC-AUC and good balance between precision and recall.
6. Results & Insights
Previous admissions and high blood sugar levels strongly correlate with readmission risk
Longer hospital stay increases risk
Number of diagnoses is a major driver
Model can support early interventions by clinicians

How to Run the Project
Install dependencies:
pip install -r requirements.txt
Run Jupyter notebook or Databricks notebook:
python patient_readmission.ipynb

📁 Project Structure
├── data/
├── notebooks/
├── src/
├── visuals/
├── README.md
└── requirements.txt

Conclusion
This project demonstrates an end-to-end machine learning workflow for predicting hospital readmissions — from cleaning data, exploring patterns, selecting the best features, training ML models, and evaluating performance.
The final Random Forest model shows strong predictive ability and can be improved with more granular patient history and hospital-level data.
