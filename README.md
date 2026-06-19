# 🤖 Customer Churn Prediction

## Project Overview

Customer Churn Prediction is a Machine Learning project that aims to predict whether a customer is likely to leave a company.

This project implements a complete Machine Learning workflow, from data preprocessing and model training to model evaluation and deployment through a Streamlit web application.

The goal is to help companies identify customers at risk of leaving and take preventive actions.

---

## Objectives

The main objectives of this project are:

- Analyze customer data
- Clean and prepare the dataset
- Transform categorical variables into numerical features
- Train multiple Machine Learning models
- Compare model performances
- Select the best performing model
- Deploy an interactive prediction application

---

## Machine Learning Workflow

```
Dataset
   |
   v
Data Cleaning
   |
   v
Feature Selection
   |
   v
Categorical Encoding
(One Hot Encoding)
   |
   v
Train / Test Split
   |
   v
Model Training
   |
   v
Model Evaluation
   |
   v
Best Model Selection
   |
   v
Streamlit Prediction App
```

---

## Project Structure

```
CUSTOMER-CHURN-PREDICTION

│
├── data
│   └── customer_churn.csv
│
├── train.py
│
├── app.py
│
├── model.pkl
│
├── model_info.txt
│
├── requirements.txt
│
└── README.md
```

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn

---

## 📊 Dataset

The dataset contains customer information such as:

- Customer demographics
- Services used
- Contract information
- Payment method
- Monthly and total charges
- Churn status

Dataset size:

```
7043 customers
```

Target variable:

```
Churn
```

Values:

```
Yes → Customer leaves
No  → Customer stays
```

---

## 🤖 Machine Learning Models

Three classification models were trained and compared:

| Model | Accuracy |
|------|----------|
| Logistic Regression | 78.46% |
| Random Forest | 78.03% |
| Decision Tree | 71.14% |

The best model selected:

```
Logistic Regression
```

The complete trained pipeline is saved as:

```
model.pkl
```

The saved pipeline contains:

- Data preprocessing
- One Hot Encoding
- Trained Machine Learning model

---

## Features

The Streamlit application provides:

###  Dataset Dashboard

Displays:

- Total number of customers
- Churn rate
- Churn distribution chart


### Model Evaluation

Shows:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix


### Customer Prediction

Users can enter customer information:

- Gender
- Senior citizen status
- Partner
- Dependents
- Tenure
- Monthly charges
- Total charges
- Contract type
- Internet service
- Payment method


The application predicts:

```
✅ Customer will stay
```

or

```
⚠️ Customer is likely to churn
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/CUSTOMER-CHURN-PREDICTION.git
```

Move into the project folder:

```bash
cd CUSTOMER-CHURN-PREDICTION
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training the Model

Run:

```bash
python train.py
```

The script will:

- Load the dataset
- Clean the data
- Train multiple models
- Compare results
- Save the best model


Output:

```
model.pkl
```

---

##  Running the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

---

##  Model Prediction Example

Example input:

```
Tenure: 12 months
Monthly Charges: 50
Contract: Month-to-month
Internet Service: DSL
```

Output:

```
Customer will stay
```

---

## Future Improvements

Possible improvements:

- Hyperparameter tuning
- Better handling of class imbalance
- More Machine Learning algorithms
- Interactive dashboards
- Model deployment online

---

## Author

Mariem