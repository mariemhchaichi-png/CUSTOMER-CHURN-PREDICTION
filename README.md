# 🤖 Customer Churn Prediction

##  Project Overview

Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to leave a company.

The project covers the complete Machine Learning workflow:

- Data loading and cleaning
- Feature preparation
- Categorical data encoding
- Model training
- Model comparison
- Model evaluation
- Model deployment using Streamlit

The objective is to help companies identify customers at risk of churn and take preventive actions.

---

## Objectives

The main goals of this project are:

- Analyze customer behavior data
- Prepare data for Machine Learning
- Train different classification algorithms
- Compare model performances
- Select the best model
- Create an interactive prediction application

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
Streamlit Application
```

---

## Project Structure

```
CUSTOMER-CHURN-PREDICTION

│
├── data/
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

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn

---

##  Dataset

The dataset contains customer information:

- Demographic information
- Services subscribed
- Contract type
- Payment method
- Monthly charges
- Total charges
- Customer churn status


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

No → Customer stays
```

---

#  Machine Learning Models

Three classification models were trained:

| Model | Accuracy |
|---|---|
| Logistic Regression | 78.46% |
| Random Forest | 78.03% |
| Decision Tree | 71.14% |

The best model selected:

```
Logistic Regression
```

The complete trained pipeline is saved in:

```
model.pkl
```

The saved pipeline contains:

- Data preprocessing
- One Hot Encoding
- Trained Machine Learning model

---

#  Model Training

To train the models:

```bash
python train.py
```

The training script:

1. Loads the dataset
2. Removes unnecessary columns
3. Converts numerical data
4. Separates features and target
5. Encodes categorical variables
6. Splits data into training and testing sets
7. Trains multiple models
8. Compares performances
9. Saves the best model

Output:

```
model.pkl
```

---

#  Streamlit Application

Run the application:

```bash
streamlit run app.py
```

The application provides:

## Dataset Dashboard

Displays:

- Number of customers
- Churn rate
- Churn distribution

## Model Evaluation

Displays:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Customer Prediction

Users can enter:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Monthly Charges
- Total Charges
- Contract
- Internet Service
- Payment Method


The application predicts:

```
✅ Customer will stay
```

or

```
⚠️ Customer is likely to churn
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/CUSTOMER-CHURN-PREDICTION.git
```

Go to the project folder:

```bash
cd CUSTOMER-CHURN-PREDICTION
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Example Prediction

Example customer:

```
Tenure: 12 months
Monthly Charges: 50
Contract: Month-to-month
Internet Service: DSL
```

Prediction:

```
Customer will stay
```

---

# Future Improvements

Possible improvements:

- Hyperparameter tuning
- Improve churn class detection
- Add more Machine Learning algorithms
- Improve UI design
- Deploy online
- Add automatic model retraining

---

# Author

Mariem