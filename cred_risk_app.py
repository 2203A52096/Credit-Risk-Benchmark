import streamlit as st
import pickle
import numpy as np
import random

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Set Streamlit page config
st.set_page_config(page_title="Credit Risk Prediction App", layout="centered")

# Sidebar removed; show navigation via buttons
page = st.radio("Navigate", ["Home", "Predict Risk", "Joke Break"])

if page == "Home":
    st.title("💳 Credit Risk Benchmark App")

    st.markdown("""
    ## Credit Risk Benchmark Problem
    Assessing the risk of a borrower defaulting on a loan is a crucial task in the finance and banking sector. Loan providers must evaluate various borrower attributes like income, credit utilization, and payment history to predict whether a customer is likely to repay or default within the next 2 years.

    The **Credit Risk Benchmark Dataset** from [Kaggle](https://www.kaggle.com/datasets/adilshamim8/credit-risk-benchmark-dataset) is used for this purpose. It includes:
    - **Financial attributes** like `monthly income`, `debt ratio`, and `revolving utilization`
    - **Demographics** such as `age` and `dependents`
    - **Credit behavior**, including late payment counts and open credit lines

    The target variable is `dlq_2yrs`:
    - `0`: No serious delinquency
    - `1`: Serious delinquency (default) within 2 years

    ## Our Solution
    We built a robust machine learning solution to tackle this problem with the following workflow:

    1. **Data Loading and Cleaning**: 
       - Missing values handled
       - Outliers removed for better data quality

    2. **Exploratory Data Analysis & Visualization**:
       - Box plots and histograms to understand feature distribution
       - Feature importance identified using advanced models

    3. **Imbalanced Class Handling**:
       - Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the target classes

    4. **Model Training**:
       - Multiple classifiers were evaluated: `SVM`, `Random Forest`, `XGBoost`, `LGBMClassifier`
       - Confusion matrix and classification reports generated

    5. **Final Prediction**:
       - Best-performing model saved and used to predict credit risk on new borrower data

    Use the **Predict Risk** tab to test the model and **Joke Break** to relax with finance-themed humor!
    """)

elif page == "Predict Risk":
    st.title("📊 Credit Risk Prediction")

    st.markdown("Fill out the borrower details below to predict default risk.")

    with st.form("prediction_form"):
        rev_util = st.number_input("Revolving Utilization (0 - 22000)", min_value=0.0, max_value=22000.0, value=443.08)
        debt_ratio = st.number_input("Debt Ratio (0 - 61106.5)", min_value=0.0, max_value=61106.5, value=322.299)
        monthly_inc = st.number_input("Monthly Income ($)", min_value=0.0, max_value=250000.0, value=5000.0)
        age = st.slider("Age", 21, 101, 48)
        late_30_59 = st.slider("Late Payments (30-59 days)", 0, 98, 0)
        open_credit = st.slider("Open Credit Lines", 0, 57, 8)
        late_90 = st.slider("Late Payments (90+ days)", 0, 98, 0)
        late_60_89 = st.slider("Late Payments (60-89 days)", 0, 98, 0)
        dependents = st.slider("Number of Dependents", 0, 8, 0)
        real_estate = st.slider("Real Estate Loans", 0, 29, 1)

        submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        features = np.array([[rev_util, debt_ratio, monthly_inc, age,
                              late_30_59, open_credit, late_90,
                              late_60_89, dependents, real_estate]])
        prediction = model.predict(features)[0]

        if prediction == 1:
            st.error("❌ Prediction: The borrower is likely to **default**.")
        else:
            st.success("✅ Prediction: The borrower is likely to **repay** the loan.")

elif page == "Joke Break":
    st.title("🤣 Money & Loan Jokes")
    jokes = [
        "Why did the banker switch careers? He lost interest.",
        "I applied for a loan to start a bakery. Now I'm rolling in dough!",
        "Why don’t banks trust atoms? They make up everything!",
        "I bought a house with a 100-year mortgage. My grandkids will love it!",
        "Why did the credit card go to therapy? It couldn’t deal with the charges.",
        "My wallet is like an onion — opening it makes me cry.",
        "I told my loan officer I’m broke. He said, 'Join the club!'"," 
        "I wanted a loan for a boat. They said I was already underwater.",
        "The ATM and I have a love-hate relationship — I love cash, it hates giving it."
    ]
    st.markdown(f"**💬 {random.choice(jokes)}**")
