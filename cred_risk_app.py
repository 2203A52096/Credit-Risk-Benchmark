import streamlit as st
import pickle
import numpy as np
import random

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Set Streamlit page config
st.set_page_config(page_title="Credit Risk Prediction App", layout="centered")

# Sidebar navigation
st.sidebar.title("📚 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "🔍 Predict Risk", "🤣 Joke Break"])

if page == "🏠 Home":
    st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #3a86ff;'>💳 Credit Risk Benchmark App</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🏦 Credit Risk Benchmark Problem
    Predicting the risk of a borrower defaulting on a loan is a cornerstone of financial decision-making. This project uses the **Credit Risk Benchmark Dataset** from [Kaggle](https://www.kaggle.com/datasets/adilshamim8/credit-risk-benchmark-dataset) to identify whether a customer is likely to default in the next 2 years.

    **Dataset Highlights:**
    - 📈 Financial attributes: `monthly income`, `debt ratio`, `revolving utilization`
    - 👤 Demographics: `age`, `dependents`
    - 💳 Credit behavior: Late payments, open credit lines
    - 🎯 Target: `dlq_2yrs` (1 = Default, 0 = No Default)

    ---

    ### 🛠️ Our ML Workflow
    1. **Data Loading & Cleaning**
        - Handled missing values
        - Removed outliers
    2. **Exploratory Analysis**
        - Visualized distributions and feature importance
    3. **Balancing the Classes**
        - Applied **SMOTE** for imbalance correction
    4. **Model Training**
        - Trained `SVM`, `Random Forest`, `XGBoost`, `LGBMClassifier`
        - Evaluated via confusion matrix and classification report
    5. **Deployment**
        - Model saved with `pickle` for live predictions

    👉 Use the **Predict Risk** page to test the model.
    🤣 Need a laugh? Visit the **Joke Break** page!
    """)

elif page == "🔍 Predict Risk":
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

        submit_button = st.form_submit_button(label='🚀 Predict')

    if submit_button:
        features = np.array([[rev_util, debt_ratio, monthly_inc, age,
                              late_30_59, open_credit, late_90,
                              late_60_89, dependents, real_estate]])
        prediction = model.predict(features)[0]

        if prediction == 1:
            st.error("❌ Prediction: The borrower is **likely to default**. Please review credit policy.")
        else:
            st.success("✅ Prediction: The borrower is **likely to repay** the loan. Safe to proceed.")

elif page == "🤣 Joke Break":
    st.markdown("""
    <div style='text-align: center;'>
        <h1>🤣 Money & Loan Jokes</h1>
        <p style='color: gray;'>Lighten up your day with a quick laugh!</p>
    </div>
    """, unsafe_allow_html=True)

    jokes = [
        "Why did the banker switch careers? He lost interest.",
        "I applied for a loan to start a bakery. Now I'm rolling in dough!",
        "Why don’t banks trust atoms? They make up everything!",
        "I bought a house with a 100-year mortgage. My grandkids will love it!",
        "Why did the credit card go to therapy? It couldn’t deal with the charges.",
        "My wallet is like an onion — opening it makes me cry.",
        "I told my loan officer I’m broke. He said, 'Join the club!'",
        "I wanted a loan for a boat. They said I was already underwater.",
        "The ATM and I have a love-hate relationship — I love cash, it hates giving it.",
        "Why did the money stay home? It didn’t feel like making cents today."
    ]

    if 'joke_idx' not in st.session_state:
        st.session_state.joke_idx = random.randint(0, len(jokes) - 1)

    st.markdown(f"""
    <div style='background-color: #f1f3f6; padding: 1rem; border-radius: 10px; text-align: center;'>
        <strong>💬 {jokes[st.session_state.joke_idx]}</strong>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Tell me another joke! 🤪"):
        st.session_state.joke_idx = random.randint(0, len(jokes) - 1)
