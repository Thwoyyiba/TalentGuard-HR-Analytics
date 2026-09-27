# 🏢 TalentGuard – HR Analytics & Employee Attrition Prediction

## Explainable Employee Attrition Prediction and Intelligent Retention Recommendation System

---

## 📖 Project Overview

TalentGuard is an end-to-end machine learning project designed to predict employee attrition risk and support data-driven HR decision-making.

The system uses employee and workplace information to predict the probability of employee attrition and classify employees into **Low, Medium, and High Risk** categories.

The project combines machine learning, SHAP explainability, and an intelligent recommendation system to help HR professionals understand employee attrition risk and identify potential retention actions.

---

## 🎯 Objectives

* Predict employee attrition risk
* Calculate employee attrition probability
* Classify employees into Low, Medium, and High Risk categories
* Identify important features contributing to attrition
* Provide SHAP-based prediction explanations
* Generate personalized HR retention recommendations
* Deploy the machine learning model using Flask
* Containerize the application using Docker
* Provide an interactive interface for employee risk prediction

---

## 🚀 Features

* Employee Attrition Prediction
* Attrition Probability Calculation
* Low / Medium / High Risk Classification
* Random Forest Classification
* SHAP Explainable AI
* Feature Importance Analysis
* Individual Prediction Explanation
* Personalized HR Recommendations
* Interactive Employee Prediction Interface
* Flask Model Deployment
* Docker Containerization
* GitHub Version Control

---

## 🤖 Machine Learning Model

The project uses a **Random Forest Classifier** to predict employee attrition.

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to perform classification.

The trained model is stored as:

```text
Random Forest Classifier.pkl
```

The model is stored using **Git LFS** because of its large file size.

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis (EDA)
   ↓
Feature Engineering
   ↓
Data Preprocessing
   ↓
Train / Test Split
   ↓
Random Forest Model Training
   ↓
Model Evaluation
   ↓
SHAP Explainability
   ↓
Attrition Probability
   ↓
Risk Classification
   ↓
HR Recommendations
   ↓
Flask Deployment
   ↓
Docker
```

---

## 📊 Risk Classification

The predicted attrition probability is converted into three employee risk categories:

| Attrition Probability | Risk Level     |
| --------------------- | -------------- |
| **70% or above**      | 🔴 High Risk   |
| **40% – 69.99%**      | 🟠 Medium Risk |
| **Below 40%**         | 🟢 Low Risk    |

These thresholds are used by the application to provide an easy-to-understand employee risk classification.

---

## 🔍 Explainable AI

TalentGuard integrates **SHAP (SHapley Additive exPlanations)** to explain machine learning predictions.

SHAP helps identify the employee characteristics that contribute to the predicted attrition risk.

### SHAP Components

* Feature Importance
* Global Feature Analysis
* Individual Prediction Explanation
* Positive and Negative Feature Contributions
* Employee-Level Risk Explanation

The trained SHAP explainer is stored as:

```text
shap_explainer.pkl
```

---

## 💡 HR Recommendation System

TalentGuard provides HR recommendations based on employee characteristics and identified attrition risk factors.

The recommendation system can help identify areas such as:

* Career Development
* Compensation
* Promotion Opportunities
* Work-Life Balance
* Overtime
* Employee Engagement
* Job Satisfaction
* Recognition and Rewards

The recommendation mapping is stored as:

```text
recommendation_mapping.pkl
```

The recommendations are intended to support HR analysis and should be considered together with organizational policies and employee circumstances.

---

## 💼 Application

The application allows HR users to enter employee information and receive:

* Employee Attrition Prediction
* Attrition Probability
* Risk Level
* SHAP-Based Feature Explanation
* Important Risk Factors
* HR Retention Recommendations

---

## 👤 Employee Information

The system can use employee-related information such as:

* Age
* Gender
* Department
* Job Role
* Marital Status
* Job Level
* Monthly Income
* Total Working Years
* Years at Company
* Years in Current Role
* Years Since Last Promotion
* Job Satisfaction
* Environment Satisfaction
* Work-Life Balance
* Job Involvement
* OverTime
  
---

## 📂 Project Structure

```text
TalentGuard-HR-Analytics/
│
├── Random Forest Classifier.pkl
├── shap_explainer.pkl
├── feature_names.pkl
├── recommendation_mapping.pkl
├── X_Scaled.pkl
│
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
├── .gitattributes
│
└── Dataset/
    └── employee_attrition_dataset.csv
```

> Large `.pkl` model files are managed using Git LFS.

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Random Forest
* SHAP
* Flask
* HTML/CSS
* Docker
* Git
* GitHub
* Git LFS

---

## 📂 Dataset

The dataset contains employee and workplace information used to analyze employee attrition.
Source:https://www.kaggle.com/datasets/ziya07/employee-attrition-prediction-dataset?select=employee_attrition_dataset_10000.csv
The dataset is used for:

* Exploratory Data Analysis
* Data Cleaning
* Feature Engineering
* Model Training
* Model Evaluation
* Attrition Prediction

---

## 📈 Model Artifacts

The project contains the following machine learning artifacts:

| File                           | Description                                      |
| ------------------------------ | ------------------------------------------------ |
| `Random Forest Classifier.pkl` | Trained Random Forest attrition prediction model |
| `shap_explainer.pkl`           | SHAP model explanation object                    |
| `feature_names.pkl`            | Model feature information                        |
| `recommendation_mapping.pkl`   | HR recommendation mappings                       |
| `X_Scaled.pkl`                 | Processed/scaled feature data                    |

The large model files are stored using **Git LFS**.



## 📌 Application Output

The TalentGuard application provides:

* Employee Attrition Prediction
* Attrition Probability
* Risk Classification
* SHAP Feature Explanation
* Important Attrition Factors
* HR Retention Recommendations
* Employee Risk Analysis

---

## 🔗 GitHub Repository

**TalentGuard –HR Analytics & Employee Attrition Prediction**

```text
https://github.com/Thwoyyiba/TalentGuard-HR-Analytics
```

---

## 🎓 Project Information

### Project Title

**TalentGuard – HR Analytics & Employee Attrition Prediction**

### Business Domain

**Human Resources / Operations Management**

### Project Type

**Analytics / Machine Learning / Dashboard / End-to-End ML Pipeline**

---

## 👩‍💻 Author

**Thwoyyiba Nasreen**

GitHub:

```text
https://github.com/Thwoyyiba
```


