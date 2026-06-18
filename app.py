import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt


# -------------------------
# Page configuration
# -------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🤖",
    layout="centered"
)


# -------------------------
# Load model + dataset
# -------------------------

model = pickle.load(
    open(
        "model.pkl",
        "rb"
    )
)


df = pd.read_csv(
    "data/customer_churn.csv"
)


# -------------------------
# Sidebar
# -------------------------

with st.sidebar:

    st.title("🤖 About Project")

    st.write(
        """
        Customer Churn Prediction

        This application uses
        Machine Learning to predict
        if a customer will leave.
        """
    )


    st.subheader("Model")

    st.write(
        "Logistic Regression"
    )


    st.subheader("Accuracy")

    st.write(
        "78.5%"
    )


    st.subheader("Technologies")

    st.write(
        """
        - Python
        - Pandas
        - Scikit-learn
        - Streamlit
        """
    )


# -------------------------
# Dataset Overview
# -------------------------

st.title(
    "🤖 Customer Churn Prediction"
)


st.subheader(
    "📊 Dataset Overview"
)


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Total Customers",
        df.shape[0]
    )


with col2:

    churn_rate = (
        df["Churn"]
        .value_counts(normalize=True)
        ["Yes"]
        * 100
    )

    st.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )



# -------------------------
# Churn Chart
# -------------------------

st.subheader(
    "📈 Churn Distribution"
)


fig, ax = plt.subplots()


df["Churn"].value_counts().plot(
    kind="bar",
    ax=ax
)


ax.set_xlabel(
    "Customer Status"
)

ax.set_ylabel(
    "Number of Customers"
)


st.pyplot(fig)



st.divider()


st.write(
    "Predict if a customer is likely to leave the company."
)


# -------------------------
# Customer information
# -------------------------

st.subheader(
    "👤 Customer Information"
)


gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)


senior = st.selectbox(
    "Senior Citizen",
    [0, 1]
)


partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)


dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)


tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)


monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=50.0
)


total = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=500.0
)


contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)


internet = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)


payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)



st.divider()


# -------------------------
# Prediction
# -------------------------

if st.button(
    "🔮 Predict Churn"
):


    data = pd.DataFrame(
        {
        "gender":[gender],
        "SeniorCitizen":[senior],
        "Partner":[partner],
        "Dependents":[dependents],
        "tenure":[tenure],
        "MonthlyCharges":[monthly],
        "TotalCharges":[total],
        "Contract":[contract],
        "InternetService":[internet],
        "PaymentMethod":[payment],

        "PhoneService":["Yes"],
        "MultipleLines":["No"],
        "OnlineSecurity":["No"],
        "OnlineBackup":["No"],
        "DeviceProtection":["No"],
        "TechSupport":["No"],
        "StreamingTV":["No"],
        "StreamingMovies":["No"],
        "PaperlessBilling":["Yes"]
        }
    )


    prediction = model.predict(data)


    probabilities = model.predict_proba(data)


    churn_probability = probabilities[0][1]

    stay_probability = probabilities[0][0]



    st.subheader(
        "📊 Prediction Result"
    )


    if prediction[0] == 1:

        st.error(
            "⚠️ Customer is likely to churn"
        )

    else:

        st.success(
            "✅ Customer will stay"
        )



    st.subheader(
        "🤖 Model Confidence"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Churn Probability",
            f"{churn_probability*100:.2f}%"
        )


    with col2:

        st.metric(
            "Stay Probability",
            f"{stay_probability*100:.2f}%"
        )



# -------------------------
# Footer
# -------------------------

st.divider()

st.caption(
    "Machine Learning Project - Customer Churn Prediction"
)