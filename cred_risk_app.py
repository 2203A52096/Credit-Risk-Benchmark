import streamlit as st
import pickle
import numpy as np
import random

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Configure page
st.set_page_config(page_title="Credit Risk App", layout="centered", page_icon="💳")

# Sidebar navigation
menu = st.sidebar.selectbox("Navigate", ["🏠 Home", "📈 Predict Risk", "🤣 Joke Break"])

if menu == "🏠 Home":
    st.title("💳 Credit Risk Benchmark App")
    st.markdown("""
    Welcome to the **Credit Risk Benchmark App**, where we assess whether a borrower is likely to default on a loan in the next two years.

    ### 🧭 Project Workflow
    - **Data Reading** from [Kaggle Dataset](https://www.kaggle.com/datasets/adilshamim8/credit-risk-benchmark-dataset)
    - **Exploration & Visualization**: Understand trends using histograms and box plots
    - **Feature Selection** and target: `dlq_2yrs`
    - **Outlier Removal** for better quality
    - **SMOTE** to balance the dataset
    - **Model Training** using SVM, Random Forest, XGBoost, LGBM
    - **Evaluation** using confusion matrix and classification report

    """)

elif menu == "📈 Predict Risk":
    st.title("📊 Credit Risk Prediction")

    st.markdown("Enter the borrower's information below:")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            rev_util = st.number_input("Revolving Utilization of Unsecured Lines", 0.0, 22000.0, step=0.1)
            debt_ratio = st.number_input("Debt Ratio", 0.0, 61106.5, step=0.1)
            monthly_inc = st.number_input("Monthly Income ($)", 0.0, 250000.0, step=100.0)
            age = st.slider("Age", 21, 101, 48)
            open_credit = st.slider("Open Credit Lines", 0, 57, 8)
        with col2:
            late_30_59 = st.slider("Late Payments (30-59 days)", 0, 98, 0)
            late_90 = st.slider("Late Payments (90+ days)", 0, 98, 0)
            late_60_89 = st.slider("Late Payments (60-89 days)", 0, 98, 0)
            real_estate = st.slider("Real Estate Loans", 0, 29, 1)
            dependents = st.slider("Number of Dependents", 0, 8, 0)

        submitted = st.form_submit_button("🔮 Predict")

    if submitted:
        features = np.array([[rev_util, debt_ratio, monthly_inc, age,
                              late_30_59, open_credit, late_90,
                              late_60_89, dependents, real_estate]])
        prediction = model.predict(features)[0]

        if prediction == 1:
            st.error("❌ Prediction: The borrower is likely to **default**.")
        else:
            st.success("✅ Prediction: The borrower is likely to **repay** the loan.")

elif menu == "🤣 Joke Break":
    st.title("🤣 Money & Loan Jokes")
    jokes = [
        "Why did the banker switch careers? He lost interest.",
        "I applied for a loan to start a bakery. Now I'm rolling in dough!",
        "Why don’t banks trust atoms? They make up everything!",
        "I bought a house with a 100-year mortgage. My grandkids will love it!",
        "Why did the credit card go to therapy? It couldn’t deal with the charges.",
        "My wallet is like an onion — opening it makes me cry.",
        "I told my loan officer I’m broke. He said, 'Join the club!'"
    ]
    st.markdown(f"**💬 {random.choice(jokes)}**")
    st.image("https://media.giphy.com/media/xT9IgG50Fb7Mi0prBC/giphy.gif", width=400)
    st.caption("Take a break — you've earned it 😎")
