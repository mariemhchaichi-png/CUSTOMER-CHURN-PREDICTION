import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

import pickle



# -------------------------
# Load dataset
# -------------------------

df = pd.read_csv(
    "data/customer_churn.csv"
)



# Remove customer ID

df = df.drop(
    "customerID",
    axis=1
)



# Convert TotalCharges

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)



df = df.dropna()



# -------------------------
# Features / Target
# -------------------------

X = df.drop(
    "Churn",
    axis=1
)


y = df["Churn"]



y = y.map(
    {
        "Yes":1,
        "No":0
    }
)



# -------------------------
# Columns
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
# Split data
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



best_model = None
best_score = 0



# -------------------------
# Training
# -------------------------

for name, model in models.items():


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


    pipeline.fit(
        X_train,
        y_train
    )


    prediction = pipeline.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        prediction
    )


    print(
        "\n",
        name
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



    if accuracy > best_score:

        best_score = accuracy

        best_model = pipeline



# -------------------------
# Save best model
# -------------------------

pickle.dump(
    best_model,
    open(
        "model.pkl",
        "wb"
    )
)



print(
    "\nBest model saved!"
)


print(
    "Best accuracy:",
    best_score
)