import pandas as pd
import pickle


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier



# -------------------------
# Load dataset
# -------------------------

df = pd.read_csv(
    "data/customer_churn.csv"
)



# -------------------------
# Data Cleaning
# -------------------------

# Remove customer ID

df = df.drop(
    "customerID",
    axis=1
)



# Convert TotalCharges to numeric

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)



# Remove missing values

df = df.dropna()



# -------------------------
# Features / Target
# -------------------------

X = df.drop(
    "Churn",
    axis=1
)


y = df["Churn"]



# Convert Yes/No to 1/0

y = y.map(
    {
        "Yes": 1,
        "No": 0
    }
)



# -------------------------
# Columns definition
# -------------------------

numeric_features = [

    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"

]


categorical_features = [

    col for col in X.columns
    if col not in numeric_features

]



# -------------------------
# Preprocessing
# -------------------------

preprocessor = ColumnTransformer(

    transformers=[

        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )

    ],

    remainder="passthrough"

)



# -------------------------
# Models
# -------------------------

models = {


    "Logistic Regression":

        LogisticRegression(
            max_iter=1000
        ),



    "Random Forest":

        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),



    "Decision Tree":

        DecisionTreeClassifier(
            random_state=42
        )

}



# -------------------------
# Train/Test split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42

)



best_model = None
best_score = 0



best_name = ""
best_precision = 0
best_recall = 0
best_f1 = 0



# -------------------------
# Training
# -------------------------

for name, model in models.items():


    print("\nTraining:", name)



    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )

        ]

    )


    # Model learning here

    pipeline.fit(

        X_train,

        y_train

    )



    # Prediction on test data

    prediction = pipeline.predict(

        X_test

    )



    # Metrics

    accuracy = accuracy_score(

        y_test,

        prediction

    )


    precision = precision_score(

        y_test,

        prediction

    )


    recall = recall_score(

        y_test,

        prediction

    )


    f1 = f1_score(

        y_test,

        prediction

    )



    print(

        "Accuracy:",

        accuracy

    )


    print(

        classification_report(

            y_test,

            prediction

        )

    )



    # Save best model

    if accuracy > best_score:


        best_score = accuracy

        best_model = pipeline

        best_name = name

        best_precision = precision

        best_recall = recall

        best_f1 = f1





# -------------------------
# Save model
# -------------------------

pickle.dump(

    best_model,

    open(

        "model.pkl",

        "wb"

    )

)



# -------------------------
# Save model information
# -------------------------

with open(

    "model_info.txt",

    "w"

) as file:


    file.write(

f"""
Best Model:
{best_name}


Accuracy:
{best_score*100:.2f}%


Precision:
{best_precision*100:.2f}%


Recall:
{best_recall*100:.2f}%


F1 Score:
{best_f1*100:.2f}%


Tested Models:

- Logistic Regression
- Random Forest
- Decision Tree

"""

    )



print("\nBest model saved!")

print(
    "Best Model:",
    best_name
)


print(
    "Best Accuracy:",
    best_score
)