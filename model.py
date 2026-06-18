import pandas as pd


def load_data():

    df = pd.read_csv(
        "data/customer_churn.csv"
    )


    # Convert TotalCharges
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )


    # Remove missing values
    df = df.dropna()


    # Remove ID
    df = df.drop(
        "customerID",
        axis=1
    )


    # Encode target
    df["Churn"] = df["Churn"].map(
        {
            "Yes": 1,
            "No": 0
        }
    )


    # Encode categorical columns
    df = pd.get_dummies(
        df,
        drop_first=True
    )


    return df