import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# Load data

df = pd.read_csv(
    "data/customer_churn.csv"
)


# Cleaning

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna()


df = df.drop(
    "customerID",
    axis=1
)



X = df.drop(
    "Churn",
    axis=1
)


y = df["Churn"].map(
    {
        "Yes":1,
        "No":0
    }
)



# Separate columns

categorical = X.select_dtypes(
    include="object"
).columns


numeric = X.select_dtypes(
    exclude="object"
).columns



# Preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical
        )
    ],
    remainder="passthrough"
)



# Model

model = LogisticRegression(
    max_iter=1000
)



from sklearn.pipeline import Pipeline


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



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



pipeline.fit(
    X_train,
    y_train
)



accuracy = pipeline.score(
    X_test,
    y_test
)


print(
    "Accuracy:",
    accuracy
)



# Save complete pipeline

pickle.dump(
    pipeline,
    open(
        "model.pkl",
        "wb"
    )
)


print(
    "Pipeline saved!"
)